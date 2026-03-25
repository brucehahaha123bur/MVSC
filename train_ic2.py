import argparse
import json
import math
import os
import random
import time
import glob
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import ConcatDataset, DataLoader

from data.harmony4d_mvsc import Harmony4DMVSCDataset

from net import encoder2 as enc_mod
from net import decoder2 as dec_mod

# NOTE:
# This trainer is adapted for the encoder2.py / decoder2.py pipeline,
# where the commonality encoder compresses both temporal redundancy and
# inter-view redundancy before JSCC coding.


# Shortened defaults.
DEFAULT_HARMONY_TRAIN_ROOT = "/root/autodl-tmp/Harmony4D/train"
DEFAULT_HARMONY_VAL_ROOT = "/root/autodl-tmp/Harmony4D/test"
DEFAULT_HARMONY_EXO_ROOT = DEFAULT_HARMONY_TRAIN_ROOT
DEFAULT_HARMONY_VAL_EXO_ROOT = DEFAULT_HARMONY_VAL_ROOT


try:
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast as _autocast
    _HAS_NEW_AMP = True
except ImportError:
    from torch.cuda.amp import GradScaler as _GradScaler
    from torch.cuda.amp import autocast as _autocast
    _HAS_NEW_AMP = False


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MVSCNet on Harmony4D-style multi-view GOP data")

    parser.add_argument(
        "--train-root",
        type=str,
        default=DEFAULT_HARMONY_EXO_ROOT,
        help="Path to one exo folder, or a higher-level parent directory (for example E:/Harmony4D/train) containing multiple */exo folders",
    )
    parser.add_argument(
        "--val-root",
        type=str,
        default=DEFAULT_HARMONY_VAL_EXO_ROOT,
        help="Path to one exo folder, or a higher-level parent directory (for example E:/Harmony4D/test) containing multiple */exo folders",
    )
    parser.add_argument("--output-dir", type=str, default="runs/mvsc_mse_scratch_snr15", help="Directory for logs/checkpoints")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--resize-shorter-to", type=int, default=0)
    parser.add_argument("--train-repeat", type=int, default=2000)
    parser.add_argument("--val-repeat", type=int, default=100)
    parser.add_argument("--min-common-frames", type=int, default=8)

    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--individual-depths", type=str, default="1,2,1")
    parser.add_argument("--common-depths", type=str, default="2,1")
    parser.add_argument("--common-heads", type=str, default="8,10")

    parser.add_argument("--channel-type", type=str, default="awgn", choices=["awgn", "rayleigh", "none"])
    parser.add_argument("--multiple-snr", type=str, default="15", help="Comma-separated SNR list, e.g. 1,4,7,10")
    parser.add_argument("--cbr-weight", type=float, default=0.0, help="Weight for CBR term in total loss")
    parser.add_argument(
        "--cbr-bits-per-component",
        type=float,
        default=3.0,
        help="Bit depth per transmitted IQ component used by CBR accounting (default: 3.0)",
    )
    parser.add_argument("--distortion-metric", type=str, default="MSE", choices=["MSE", "SSIM", "MS-SSIM"])

    parser.add_argument(
        "--train-stage",
        type=str,
        default="full",
        choices=["full", "ic_only", "individual_only"],
        help=(
            "Training stage selector: full uses the entire MVSC pipeline; "
            "ic_only trains individual+commonality encoder/decoder only; "
            "individual_only trains only the individual encoder/decoder."
        ),
    )
    parser.add_argument(
        "--freeze-individual",
        action="store_true",
        help="Freeze the individual encoder/decoder parameters during training.",
    )
    parser.add_argument(
        "--freeze-commonality",
        action="store_true",
        help="Freeze the commonality encoder/decoder parameters during training.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional checkpoint path used to preload model weights before training.",
    )

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable automatic mixed precision on CUDA")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision on CUDA")
    parser.set_defaults(amp=True)
    # max-nonfinite-batches removed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1)
    # val-interval-steps, early stop args removed

    return parser.parse_args()


def resolve_device(device_name: str):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def parse_root_list(root_spec: str):
    roots = []
    for token in str(root_spec).split(","):
        token = token.strip()
        if token:
            roots.append(os.path.abspath(token))
    if not roots:
        raise ValueError("Dataset root specification is empty.")
    return roots


def split_repeat_budget(total_repeat: int, num_parts: int):
    total_repeat = int(total_repeat)
    num_parts = int(num_parts)
    if total_repeat <= 0:
        raise ValueError("repeat budget must be > 0")
    if num_parts <= 0:
        raise ValueError("num_parts must be > 0")

    base = total_repeat // num_parts
    rem = total_repeat % num_parts
    out = []
    for i in range(num_parts):
        out.append(base + (1 if i < rem else 0))
    return out

def build_dataset(root, args, is_train: bool):
    resize_shorter_to = args.resize_shorter_to if args.resize_shorter_to > 0 else None
    split_name = "Train" if is_train else "Val"
    root_list = parse_root_list(root)
    repeat_budget = int(args.train_repeat if is_train else args.val_repeat)

    for r in root_list:
        if not os.path.isdir(r):
            raise FileNotFoundError(f"Dataset root does not exist: {r}")

    if len(root_list) == 1:
        root_abs = root_list[0]
        print(f"[Info] {split_name} uses root: {root_abs}")
        print(f"[Info] {split_name} repeat budget: {repeat_budget}")
        return Harmony4DMVSCDataset(
            root=root_abs,
            num_views=args.num_views,
            num_frames=args.num_frames,
            crop_size=args.crop_size,
            resize_shorter_to=resize_shorter_to,
            random_crop=is_train,
            random_flip=is_train,
            min_common_frames=args.min_common_frames,
            repeat=repeat_budget,
        )

    per_root_repeats = split_repeat_budget(repeat_budget, len(root_list))
    print(f"[Info] {split_name} uses {len(root_list)} roots:")
    for i, (r, rep) in enumerate(zip(root_list, per_root_repeats), start=1):
        print(f"[Info]   [{i}] root={r} repeat={rep}")

    datasets = []
    for r, rep in zip(root_list, per_root_repeats):
        datasets.append(
            Harmony4DMVSCDataset(
                root=r,
                num_views=args.num_views,
                num_frames=args.num_frames,
                crop_size=args.crop_size,
                resize_shorter_to=resize_shorter_to,
                random_crop=is_train,
                random_flip=is_train,
                min_common_frames=args.min_common_frames,
                repeat=rep,
            )
        )

    return ConcatDataset(datasets)


def build_loader(dataset, args, is_train: bool, device):
    return DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else 1,
        shuffle=is_train,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=is_train,
    )


def make_grad_scaler(enable_amp: bool, device):
    if not (enable_amp and device.type == "cuda"):
        return None
    if _HAS_NEW_AMP:
        return _GradScaler("cuda", enabled=True)
    return _GradScaler(enabled=True)


def autocast_context(enable_amp: bool, device):
    if not (enable_amp and device.type == "cuda"):
        return nullcontext()
    if _HAS_NEW_AMP:
        return _autocast("cuda", enabled=True)
    return _autocast(enabled=True)


def make_model_args(args):
    return args




def compute_psnr(x_hat, x):
    mse = torch.mean((x_hat - x) ** 2).item()
    mse = max(mse, 1e-12)
    return -10.0 * math.log10(mse)


def parse_multiple_snr(multiple_snr: str):
    values = []
    for token in str(multiple_snr).split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("--multiple-snr must contain at least one numeric value.")
    return values

def parse_int_tuple(spec: str, expected_len: int, name: str):
    values = []
    for token in str(spec).split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if len(values) != expected_len:
        raise ValueError(f"{name} must contain exactly {expected_len} integers, got: {spec}")
    return tuple(values)


class ModernMVSCNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_views = int(args.num_views)
        self.crop_size = int(args.crop_size)
        self.embed_dim = int(args.embed_dim)
        self.token_dim = 4 * self.embed_dim
        self.latent_dim = int(args.latent_dim)
        self.individual_depths = parse_int_tuple(args.individual_depths, 3, "--individual-depths")
        self.common_depths = parse_int_tuple(args.common_depths, 2, "--common-depths")
        self.common_heads = parse_int_tuple(args.common_heads, 2, "--common-heads")
        self.channel_type = str(args.channel_type)
        self.snr_values = parse_multiple_snr(args.multiple_snr)
        self.cbr_weight = float(args.cbr_weight)
        self.cbr_bits_per_component = float(getattr(args, "cbr_bits_per_component", 3.0))
        self.distortion_metric = str(args.distortion_metric).upper()
        self.train_stage = str(getattr(args, "train_stage", "full"))
        self.freeze_individual = bool(getattr(args, "freeze_individual", False))
        self.freeze_commonality = bool(getattr(args, "freeze_commonality", False))

        if self.distortion_metric != "MSE":
            raise ValueError(
                "ModernMVSCNet currently supports --distortion-metric MSE only. "
                f"Got: {args.distortion_metric}"
            )

        self.individual_encoder = enc_mod.MVSC_Individual_Encoder(
            img_size=self.crop_size,
            patch_size=2,
            in_chans=3,
            embed_dim=self.embed_dim,
            depths=self.individual_depths,
        )

        # encoder2 individual encoder outputs 192 channels.
        self.common_encoder = enc_mod.MVSC_Commonality_Encoder(
            dim=192,
            input_resolution=(self.crop_size // 8, self.crop_size // 8),
            depths=self.common_depths,
            num_heads=self.common_heads,
        )

        # encoder2 commonality encoder outputs [B, T/2, V/2, 1024, 320].
        self.compressed_num_views = max(1, self.num_views // 2)

        self.jscc_encoder = enc_mod.MVSC_JSCC_Encoder(
            dim=320,
            latent_dim=self.latent_dim,
        )

        self.jscc_decoder = dec_mod.MVSC_JSCC_Decoder(
            latent_dim=self.latent_dim,
            embed_dim=320,
            compressed_num_views=self.compressed_num_views,
            temporal_upsample_in_jscc=False,
        )

        # decoder2 mirrors encoder2: first decode compressed common tokens,
        # then restore V and T inside the commonality decoder.
        self.common_decoder = dec_mod.MVSC_Commonality_Decoder(
            dim=320,
            input_resolution=(self.crop_size // 8, self.crop_size // 8),
            num_views=self.num_views,
            compressed_num_views=self.compressed_num_views,
            depths=tuple(reversed(self.common_depths)),
            num_heads=tuple(reversed(self.common_heads)),
        )

        # decoder2 self-test uses patch_size=8 with input_resolution=H/8,W/8,
        # so keep the no-extra-upsampling reconstruction path here.
        self.individual_decoder = dec_mod.MVSC_Individual_Decoder(
            img_size=self.crop_size,
            patch_size=8,
            out_chans=3,
            embed_dim=192,
            input_resolution=(self.crop_size // 8, self.crop_size // 8),
            num_upsample_stages=0,
            depths=self.individual_depths,
        )

        if self.freeze_individual:
            self._set_requires_grad(self.individual_encoder, False)
            self._set_requires_grad(self.individual_decoder, False)

        if self.freeze_commonality:
            self._set_requires_grad(self.common_encoder, False)
            self._set_requires_grad(self.common_decoder, False)


    @staticmethod
    def _set_requires_grad(module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def _sample_snr(self, given_SNR=None):
        if given_SNR is not None:
            return float(given_SNR)
        return float(random.choice(self.snr_values))

    def _power_normalize(self, y):
        power = torch.mean(y.pow(2), dim=tuple(range(1, y.ndim)), keepdim=True)
        return y / torch.sqrt(power + 1e-8)

    def _apply_channel(self, y, snr_db: float):
        if self.channel_type == "none":
            return y

        y = self._power_normalize(y)
        noise_std = 10.0 ** (-float(snr_db) / 20.0)

        if self.channel_type == "awgn":
            noise = torch.randn_like(y) * noise_std
            return y + noise

        if self.channel_type == "rayleigh":
            h = torch.randn(y.shape[0], 1, 1, 1, device=y.device, dtype=y.dtype)
            h = torch.clamp(h.abs(), min=1e-3)
            noise = torch.randn_like(y) * noise_std
            return (h * y + noise) / h

        raise ValueError(f"Unsupported channel_type: {self.channel_type}")

    def _compute_cbr(self, y, x):
        bits = y.shape[1] * y.shape[2] * y.shape[3] * self.cbr_bits_per_component
        denom = x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4] * x.shape[5]
        value = float(bits) / float(denom)
        return x.new_tensor(value)

    def forward(self, x, given_SNR=None):
        x_ind = self.individual_encoder(x)

        if self.train_stage == "individual_only":
            x_hat = self.individual_decoder(x_ind)
            used_snr = None
            distortion = torch.mean((x_hat - x) ** 2)
            cbr = x.new_tensor(0.0)
            loss = distortion
            aux = {
                "distortion": distortion.detach(),
                "cbr": cbr.detach(),
                "stage": self.train_stage,
            }
            return x_hat, used_snr, loss, aux

        x_com = self.common_encoder(x_ind)
        x_com_dec = x_com
        used_snr = None
        cbr = x.new_tensor(0.0)

        # Full-stage JSCC path: encode, channel, decode commonality tokens.
        if self.train_stage == "full":
            y = self.jscc_encoder(x_com)
            used_snr = self._sample_snr(given_SNR)
            y_hat = self._apply_channel(y, used_snr)
            x_com_dec = self.jscc_decoder(y_hat)
            cbr = self._compute_cbr(y, x)
        elif self.train_stage != "ic_only":
            raise ValueError(f"Unsupported train_stage: {self.train_stage}")

        x_com_dec = self.common_decoder(x_com_dec)
        x_hat = self.individual_decoder(x_com_dec)

        distortion = torch.mean((x_hat - x) ** 2)
        loss = distortion + self.cbr_weight * cbr
        aux = {
            "distortion": distortion.detach(),
            "cbr": cbr.detach(),
            "stage": self.train_stage,
        }
        return x_hat, used_snr, loss, aux


def resolve_epoch_snr(epoch: int, args, snr_values):
    raise NotImplementedError("resolve_epoch_snr was removed in the simplified trainer.")


def resolve_epoch_cbr_weight(epoch: int, args):
    raise NotImplementedError("resolve_epoch_cbr_weight was removed in the simplified trainer.")


def is_metric_improved(metric_name: str, current: float, best: float, min_delta: float):
    raise NotImplementedError("is_metric_improved was removed in the simplified trainer.")


def _flatten_string_fields(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_string_fields(item))
        return out
    return []


def _batch_debug_desc(batch):
    cams = _flatten_string_fields(batch.get("cam_names", []))
    frames = _flatten_string_fields(batch.get("frame_names", []))
    cams_text = ",".join(cams[:8]) if cams else "n/a"
    frames_text = ",".join(frames[:8]) if frames else "n/a"
    return f"cams=[{cams_text}] frames=[{frames_text}]"


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    args,
    epoch,
    given_snr_override=None,
    global_step=0,
    optimizer_step=0,
    val_interval_steps=0,
    on_val_interval=None,
):
    model.train()
    loss_meter = AverageMeter()
    distortion_meter = AverageMeter()
    cbr_meter = AverageMeter()
    psnr_meter = AverageMeter()
    nonfinite_skip_count = 0

    start = time.time()
    for step, batch in enumerate(loader, start=1):
        global_step += 1
        x = batch["x"].to(device, non_blocking=True)

        if not torch.isfinite(x).all():
            nonfinite_skip_count += 1
            print(
                f"[Warn] Non-finite input detected at epoch={epoch} step={step}. "
                f"skip_count={nonfinite_skip_count} {_batch_debug_desc(batch)}"
            )
            continue

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(args.amp, device):
            model_out = model(x, given_SNR=given_snr_override)

        if isinstance(model_out, tuple) and len(model_out) == 4:
            x_hat, used_snr, loss, aux = model_out
            distortion_value = float(aux["distortion"].item()) if "distortion" in aux else float(loss.item())
            cbr_value = float(aux["cbr"].item()) if "cbr" in aux else 0.0
        else:
            x_hat, used_snr, loss = model_out
            distortion_value = float(loss.item())
            cbr_value = 0.0

        loss_is_finite = bool(torch.isfinite(loss).all().item())
        x_hat_is_finite = bool(torch.isfinite(x_hat).all().item())
        if not (loss_is_finite and x_hat_is_finite):
            nonfinite_skip_count += 1
            loss_text = float(loss.detach().float().mean().item()) if torch.numel(loss) > 0 else float("nan")
            print(
                f"[Warn] Non-finite forward detected at epoch={epoch} step={step} "
                f"loss={loss_text:.6f} snr={used_snr} skip_count={nonfinite_skip_count} "
                f"{_batch_debug_desc(batch)}"
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                # Advance scaler state on skipped step to keep dynamic scaling responsive.
                scaler.update()
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        optimizer_step += 1

        bsz = x.shape[0]
        loss_meter.update(loss.item(), bsz)
        distortion_meter.update(distortion_value, bsz)
        cbr_meter.update(cbr_value, bsz)
        psnr_meter.update(compute_psnr(x_hat.detach(), x.detach()), bsz)

        if step % args.log_interval == 0 or step == len(loader):
            elapsed = time.time() - start
            print(
                f"[Train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss_meter.val:.6f} avg_loss={loss_meter.avg:.6f} "
                f"dist={distortion_meter.val:.6f} avg_dist={distortion_meter.avg:.6f} "
                f"cbr={cbr_meter.val:.6f} avg_cbr={cbr_meter.avg:.6f} "
                f"psnr={psnr_meter.val:.3f} avg_psnr={psnr_meter.avg:.3f} "
                f"snr={used_snr} time={elapsed:.1f}s"
            )

        if (
            on_val_interval is not None
            and val_interval_steps > 0
            and global_step % val_interval_steps == 0
        ):
            on_val_interval(global_step=global_step, epoch=epoch, train_step=step)
            # evaluate() switches model to eval; set back to train for next steps.
            model.train()

    if nonfinite_skip_count > 0:
        print(f"[Warn] Epoch {epoch} skipped non-finite train batches: {nonfinite_skip_count}")

    return (
        loss_meter.avg,
        psnr_meter.avg,
        distortion_meter.avg,
        cbr_meter.avg,
        global_step,
        optimizer_step,
        nonfinite_skip_count,
    )


def evaluate(model, loader, device, args, epoch, given_snr_override=None):
    model.eval()
    loss_meter = AverageMeter()
    distortion_meter = AverageMeter()
    cbr_meter = AverageMeter()
    psnr_meter = AverageMeter()
    nonfinite_skip_count = 0

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            x = batch["x"].to(device, non_blocking=True)

            if not torch.isfinite(x).all():
                nonfinite_skip_count += 1
                print(
                    f"[Warn] Non-finite input in validation at epoch={epoch} step={step}. "
                    f"skip_count={nonfinite_skip_count} {_batch_debug_desc(batch)}"
                )
                continue

            model_out = model(x, given_SNR=given_snr_override)

            if isinstance(model_out, tuple) and len(model_out) == 4:
                x_hat, used_snr, loss, aux = model_out
                distortion_value = float(aux["distortion"].item()) if "distortion" in aux else float(loss.item())
                cbr_value = float(aux["cbr"].item()) if "cbr" in aux else 0.0
            else:
                x_hat, used_snr, loss = model_out
                distortion_value = float(loss.item())
                cbr_value = 0.0

            if not (bool(torch.isfinite(loss).all().item()) and bool(torch.isfinite(x_hat).all().item())):
                nonfinite_skip_count += 1
                loss_text = float(loss.detach().float().mean().item()) if torch.numel(loss) > 0 else float("nan")
                print(
                    f"[Warn] Non-finite validation forward at epoch={epoch} step={step} "
                    f"loss={loss_text:.6f} snr={used_snr} skip_count={nonfinite_skip_count} "
                    f"{_batch_debug_desc(batch)}"
                )
                continue

            bsz = x.shape[0]
            loss_meter.update(loss.item(), bsz)
            distortion_meter.update(distortion_value, bsz)
            cbr_meter.update(cbr_value, bsz)
            psnr_meter.update(compute_psnr(x_hat, x), bsz)

            if step % args.log_interval == 0 or step == len(loader):
                print(
                    f"[Val] epoch={epoch} step={step}/{len(loader)} "
                    f"loss={loss_meter.val:.6f} avg_loss={loss_meter.avg:.6f} "
                    f"dist={distortion_meter.val:.6f} avg_dist={distortion_meter.avg:.6f} "
                    f"cbr={cbr_meter.val:.6f} avg_cbr={cbr_meter.avg:.6f} "
                    f"psnr={psnr_meter.val:.3f} avg_psnr={psnr_meter.avg:.3f} "
                    f"snr={used_snr}"
                )

    if loss_meter.count == 0:
        raise RuntimeError(
            f"Validation produced no finite batches at epoch={epoch}. "
            f"skipped={nonfinite_skip_count}"
        )

    if nonfinite_skip_count > 0:
        print(f"[Warn] Epoch {epoch} skipped non-finite val batches: {nonfinite_skip_count}")

    return loss_meter.avg, psnr_meter.avg, distortion_meter.avg, cbr_meter.avg


def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)


    # Resume logic removed by simplification


def main():
    args = parse_args()
    set_seed(args.seed)

    # snr_values = parse_multiple_snr(args.multiple_snr)

    # Only keep basic checks as per simplification
    if args.train_repeat <= 0:
        raise ValueError("--train-repeat must be > 0.")
    if args.val_repeat <= 0:
        raise ValueError("--val-repeat must be > 0.")
    if args.cbr_weight < 0:
        raise ValueError("--cbr-weight must be >= 0.")

    device = resolve_device(args.device)
    if device.type != "cuda" and args.channel_type in {"awgn", "rayleigh"}:
        print(
            "[Warn] CUDA is unavailable in current environment. "
            f"Fallback channel_type: {args.channel_type} -> none"
        )
        args.channel_type = "none"
    if args.distortion_metric == "MS-SSIM" and device.type != "cuda":
        raise ValueError("MS-SSIM in this project expects CUDA. Use --device cuda or switch to --distortion-metric MSE.")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train_dataset = build_dataset(args.train_root, args, is_train=True)
    train_loader = build_loader(train_dataset, args, is_train=True, device=device)

    val_root = args.val_root if args.val_root is not None else args.train_root
    val_dataset = build_dataset(val_root, args, is_train=False)
    val_loader = build_loader(val_dataset, args, is_train=False, device=device)


    model_args = make_model_args(args)
    model = ModernMVSCNet(model_args).to(device)
    model.channel_type = args.channel_type
    if args.pretrained is not None:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained}")
        print(f"[Info] Loading pretrained checkpoint: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        model_state = model.state_dict()
        filtered_state = {}
        skipped_mismatch = []
        skipped_unknown = []
        for k, v in state_dict.items():
            if k not in model_state:
                skipped_unknown.append(k)
                continue
            if model_state[k].shape != v.shape:
                skipped_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                continue
            filtered_state[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        print(
            f"[Info] Pretrained load done. loaded_keys={len(filtered_state)} "
            f"missing_keys={len(missing_keys)} unexpected_keys={len(unexpected_keys)} "
            f"skipped_mismatch={len(skipped_mismatch)} skipped_unknown={len(skipped_unknown)}"
        )
        if len(skipped_mismatch) > 0:
            print("[Info] Skipped mismatched keys:")
            for name, ckpt_shape, model_shape in skipped_mismatch:
                print(f"  - {name}: ckpt{ckpt_shape} != model{model_shape}")
        if len(skipped_unknown) > 0:
            print(f"[Info] Skipped unknown keys: {skipped_unknown}")
        if len(missing_keys) > 0:
            print(f"[Info] Missing keys after filtered load: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"[Info] Unexpected keys after filtered load: {unexpected_keys}")

    if args.train_stage != "full":
        if args.channel_type != "none":
            print(
                f"[Info] train_stage={args.train_stage} disables channel usage. "
                f"Override channel_type: {args.channel_type} -> none"
            )
        model.channel_type = "none"

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after applying freeze options.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = make_grad_scaler(args.amp, device)

    # Initialize training state
    start_epoch = 1
    global_step = 0
    optimizer_step = 0
    best_val_loss = float("inf")
    best_val_psnr = float("-inf")

    if args.pretrained is None:
        print("[Info] Training from scratch.")
    else:
        print("[Info] Training starts from pretrained weights.")

    print(
        f"[Info] Start state: epoch={start_epoch}, global_step={global_step}, optimizer_step={optimizer_step}"
    )

    if start_epoch > args.epochs:
        print(
            f"[Info] Start epoch {start_epoch} is already beyond target epochs {args.epochs}. "
            "Nothing to train."
        )
        return

    def build_state(epoch_idx, step_idx):
        return {
            "epoch": epoch_idx,
            "global_step": step_idx,
            "optimizer_step": optimizer_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val_loss,
            "best_val_psnr": best_val_psnr,
            "args": vars(args),
        }

    def maybe_save_best(val_loss, epoch_idx, step_idx, source):
        nonlocal best_val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = build_state(epoch_idx, step_idx)
            state["best_val_loss"] = best_val_loss
            save_checkpoint(state, os.path.join(args.output_dir, "best.pt"))
            print(f"Saved new best checkpoint ({source}) with val_loss={best_val_loss:.6f}")

    def maybe_save_best_psnr(val_psnr, epoch_idx, step_idx, source):
        nonlocal best_val_psnr
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            state = build_state(epoch_idx, step_idx)
            state["best_val_psnr"] = best_val_psnr
            save_checkpoint(state, os.path.join(args.output_dir, "best_psnr.pt"))
            print(f"Saved new best-psnr checkpoint ({source}) with val_psnr={best_val_psnr:.3f}")

    # Interval validation and related bookkeeping removed by simplification

    print(
        f"[Info] Model config: embed_dim={args.embed_dim}, token_dim={4 * args.embed_dim}, latent_dim={args.latent_dim}, "
        f"individual_depths={args.individual_depths}, common_depths={args.common_depths}, common_heads={args.common_heads}"
    )
    print(
        f"[Info] Train stage: {args.train_stage}, "
        f"freeze_individual={args.freeze_individual}, freeze_commonality={args.freeze_commonality}, "
        f"channel_type={args.channel_type}"
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(
        f"[Info] LR scheduler: cosine, T_max={args.epochs}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        # SNR/CBR/validation scheduling logic removed, simplified logic below:
        model.cbr_weight = float(args.cbr_weight)
        current_val_snr = None
        print(f"[Info] epoch={epoch} cbr_weight={model.cbr_weight:.3f}")

        train_loss, train_psnr, train_dist, train_cbr, global_step, optimizer_step, train_nonfinite_skips = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
            given_snr_override=None,
            global_step=global_step,
            optimizer_step=optimizer_step,
            val_interval_steps=0,
            on_val_interval=None,
        )

        val_loss, val_psnr, val_dist, val_cbr = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            args=args,
            epoch=epoch,
            given_snr_override=None,
        )

        maybe_save_best(val_loss, epoch, global_step, source="epoch-end")
        maybe_save_best_psnr(val_psnr, epoch, global_step, source="epoch-end")

        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.6f} train_dist={train_dist:.6f} train_cbr={train_cbr:.6f} train_psnr={train_psnr:.3f} "
            f"val_loss={val_loss:.6f} val_dist={val_dist:.6f} val_cbr={val_cbr:.6f} val_psnr={val_psnr:.3f} "
            f"lr={lr_now:.6e} opt_steps={optimizer_step} skipped_nonfinite_train={train_nonfinite_skips}"
        )

        latest_path = os.path.join(args.output_dir, "latest.pt")
        state = build_state(epoch, global_step)

        if epoch % args.save_every == 0:
            save_checkpoint(state, latest_path)

    print("Training finished.")


if __name__ == "__main__":
    main()
