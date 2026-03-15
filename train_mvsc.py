import argparse
import json
import math
import os
import random
import time
import glob
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from data.harmony4d_mvsc import Harmony4DMVSCDataset
from net.network import MVSCNet


DEFAULT_HARMONY_SCENE_ROOT = "E:/Harmony4D/train/01_hugging/01_hugging/001_hugging"
DEFAULT_HARMONY_EXO_ROOT = f"{DEFAULT_HARMONY_SCENE_ROOT}/exo"
DEFAULT_HARMONY_VAL_EXO_ROOT = "E:/Harmony4D/test/01_hugging/002_hugging/exo"


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
        help="Path to one exo folder, or a parent directory containing multiple */exo folders",
    )
    parser.add_argument(
        "--val-root",
        type=str,
        default=DEFAULT_HARMONY_VAL_EXO_ROOT,
        help="Path to one exo folder, or a parent directory containing multiple */exo folders",
    )
    parser.add_argument("--output-dir", type=str, default="runs/mvsc_mse_scratch_snr15", help="Directory for logs/checkpoints")
    parser.add_argument("--resume", type=str, default="runs/mvsc_mse_scratch_snr15/latest.pt", help="Path to a checkpoint (.pt) for resume training")
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="Load only model weights from --resume and reinitialize optimizer/scheduler",
    )
    parser.add_argument(
        "--resume-ignore-mismatch",
        action="store_true",
        help="Load model with strict=False when resuming (allow missing/unexpected keys)",
    )

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-step-unit",
        type=str,
        default="batch",
        choices=["batch", "epoch"],
        help="Unit for cosine LR updates: batch (faster updates) or epoch.",
    )
    parser.add_argument(
        "--lr-step-interval",
        type=int,
        default=4,
        help="Apply one LR scheduler step every N units (batch/epoch according to --lr-step-unit).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--resize-shorter-to", type=int, default=0)
    parser.add_argument("--train-repeat", type=int, default=2000)
    parser.add_argument("--val-repeat", type=int, default=100)
    parser.add_argument(
        "--repeat-per-exo",
        action="store_true",
        help=(
            "When multiple exo roots are discovered, treat --train-repeat/--val-repeat as per-exo repeat. "
            "By default repeat budget is split across exo roots to keep epoch length stable."
        ),
    )
    parser.add_argument("--min-common-frames", type=int, default=8)

    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--common-depth", type=int, default=2)
    parser.add_argument("--common-heads", type=int, default=4)

    parser.add_argument("--channel-type", type=str, default="awgn", choices=["awgn", "rayleigh", "none"])
    parser.add_argument("--multiple-snr", type=str, default="15", help="Comma-separated SNR list, e.g. 1,4,7,10")
    parser.add_argument("--given-snr", type=float, default=15.0, help="Fixed SNR for forward pass, default uses args.multiple_snr")
    parser.add_argument("--val-given-snr", type=float, default=15.0, help="Fixed SNR used for validation; default uses current phase or first value in --multiple-snr")
    parser.add_argument("--snr-warmup-epochs", type=int, default=0, help="Warmup epochs using fixed SNR before random-SNR training")
    parser.add_argument("--snr-warmup-snr", type=float, default=None, help="Fixed SNR used during warmup; default uses first value in --multiple-snr")
    parser.add_argument("--snr-finetune-epochs", type=int, default=0, help="Final epochs using fixed SNR after random-SNR training")
    parser.add_argument("--snr-finetune-snr", type=float, default=None, help="Fixed SNR used during finetune; default uses first value in --multiple-snr")
    parser.add_argument("--cbr-weight", type=float, default=0.0, help="Weight for CBR term in total loss")
    parser.add_argument(
        "--cbr-bits-per-component",
        type=float,
        default=3.0,
        help="Bit depth per transmitted IQ component used by CBR accounting (default: 3.0)",
    )
    parser.add_argument("--cbr-warmup-epochs", type=int, default=0, help="Warmup epochs using a dedicated CBR weight")
    parser.add_argument("--cbr-warmup-weight", type=float, default=None, help="CBR weight during warmup; default uses --cbr-weight")
    parser.add_argument("--cbr-finetune-epochs", type=int, default=0, help="Final epochs using a dedicated CBR weight")
    parser.add_argument("--cbr-finetune-weight", type=float, default=None, help="CBR weight during finetune; default uses --cbr-weight")
    parser.add_argument("--distortion-metric", type=str, default="MSE", choices=["MSE", "SSIM", "MS-SSIM"])

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable automatic mixed precision on CUDA")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision on CUDA")
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--max-nonfinite-batches",
        type=int,
        default=20,
        help="Maximum number of skipped non-finite batches per epoch before aborting (0 means fail on first).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--val-interval-steps",
        type=int,
        default=0,
        help="Run validation every N global train steps. 0 disables mid-epoch validation.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop when monitored validation metric does not improve for N epochs. 0 disables.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum metric improvement required to reset early-stop counter.",
    )
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="psnr",
        choices=["psnr", "loss"],
        help="Validation metric used by early stopping.",
    )

    return parser.parse_args()


def resolve_device(device_name: str):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _is_exo_dir(path: str):
    if not os.path.isdir(path):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    for name in names:
        cam_path = os.path.join(path, name)
        if os.path.isdir(cam_path) and name.startswith("cam"):
            return True
    return False


def discover_exo_roots(root: str):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    root_abs = os.path.abspath(root)
    if _is_exo_dir(root_abs):
        return [root_abs]

    exo_roots = []
    for dirpath, dirnames, _ in os.walk(root_abs):
        if os.path.basename(dirpath).lower() == "exo" and _is_exo_dir(dirpath):
            exo_roots.append(dirpath)
            # No need to walk below an exo leaf.
            dirnames[:] = []

    exo_roots = sorted(set(exo_roots))
    if exo_roots:
        return exo_roots

    raise ValueError(
        "No valid exo folder found under dataset root. "
        "Expected one exo folder with camXX subfolders, or a parent containing */exo folders. "
        f"root={root_abs}"
    )


def _split_repeat_budget(total_repeat: int, parts: int):
    total_repeat = int(total_repeat)
    parts = int(parts)
    if total_repeat <= 0:
        raise ValueError("repeat must be > 0")
    if parts <= 0:
        raise ValueError("parts must be > 0")
    if parts == 1:
        return [total_repeat]

    base = total_repeat // parts
    rem = total_repeat % parts

    # Ensure every discovered exo contributes at least one sample.
    if base == 0:
        return [1 for _ in range(parts)]

    repeats = []
    for i in range(parts):
        repeats.append(base + (1 if i < rem else 0))
    return repeats


def build_dataset(root, args, is_train: bool):
    resize_shorter_to = args.resize_shorter_to if args.resize_shorter_to > 0 else None
    exo_roots = discover_exo_roots(root)
    repeat_budget = int(args.train_repeat if is_train else args.val_repeat)
    split_name = "Train" if is_train else "Val"

    if len(exo_roots) == 1:
        print(f"[Info] {split_name} uses exo root: {exo_roots[0]}")
        return Harmony4DMVSCDataset(
            root=exo_roots[0],
            num_views=args.num_views,
            num_frames=args.num_frames,
            crop_size=args.crop_size,
            resize_shorter_to=resize_shorter_to,
            random_crop=is_train,
            random_flip=is_train,
            min_common_frames=args.min_common_frames,
            repeat=repeat_budget,
        )

    if args.repeat_per_exo:
        per_exo_repeats = [repeat_budget for _ in exo_roots]
        repeat_desc = f"per-exo={repeat_budget}, total={sum(per_exo_repeats)}"
    else:
        per_exo_repeats = _split_repeat_budget(repeat_budget, len(exo_roots))
        repeat_desc = f"split-budget={repeat_budget}, allocated_total={sum(per_exo_repeats)}"

    print(f"[Info] {split_name} discovered {len(exo_roots)} exo roots under: {os.path.abspath(root)}")
    print(f"[Info] {split_name} repeat policy: {repeat_desc}")
    preview = exo_roots[:8]
    for i, exo_root in enumerate(preview, start=1):
        print(f"[Info]   [{i}] {exo_root}")
    if len(exo_roots) > len(preview):
        print(f"[Info]   ... ({len(exo_roots) - len(preview)} more)")

    datasets = []
    for exo_root, repeat in zip(exo_roots, per_exo_repeats):
        datasets.append(
            Harmony4DMVSCDataset(
                root=exo_root,
                num_views=args.num_views,
                num_frames=args.num_frames,
                crop_size=args.crop_size,
                resize_shorter_to=resize_shorter_to,
                random_crop=is_train,
                random_flip=is_train,
                min_common_frames=args.min_common_frames,
                repeat=repeat,
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
    return SimpleNamespace(
        channel_type=args.channel_type,
        multiple_snr=args.multiple_snr,
        cbr_weight=args.cbr_weight,
        cbr_bits_per_component=float(getattr(args, "cbr_bits_per_component", 3.0)),
        distortion_metric=args.distortion_metric,
        trainset="MVSC",
    )


def make_config(args, device):
    return SimpleNamespace(
        norm=False,
        device=device,
        CUDA=device.type == "cuda",
        logger=None,
        image_dims=(3, args.crop_size, args.crop_size),
        mvsc_patch_size=args.patch_size,
        mvsc_embed_dim=args.embed_dim,
        mvsc_latent_dim=args.latent_dim,
        mvsc_num_views=args.num_views,
        mvsc_common_depth=args.common_depth,
        mvsc_common_heads=args.common_heads,
    )


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


def resolve_epoch_snr(epoch: int, args, snr_values):
    """
    Returns:
        train_snr: float or None (None means random sample in model forward)
        val_snr: float (fixed for stable validation)
        phase_name: str for logging
    """
    if args.given_snr is not None:
        train_snr = float(args.given_snr)
        val_snr = float(args.val_given_snr) if args.val_given_snr is not None else train_snr
        return train_snr, val_snr, "fixed"

    warmup_epochs = int(max(args.snr_warmup_epochs, 0))
    finetune_epochs = int(max(args.snr_finetune_epochs, 0))

    warmup_snr = float(args.snr_warmup_snr) if args.snr_warmup_snr is not None else float(snr_values[0])
    finetune_snr = float(args.snr_finetune_snr) if args.snr_finetune_snr is not None else float(snr_values[0])

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        train_snr = warmup_snr
        phase_name = "warmup-fixed"
    elif finetune_epochs > 0 and epoch > args.epochs - finetune_epochs:
        train_snr = finetune_snr
        phase_name = "finetune-fixed"
    else:
        train_snr = None
        phase_name = "random"

    if args.val_given_snr is not None:
        val_snr = float(args.val_given_snr)
    elif train_snr is not None:
        val_snr = train_snr
    else:
        # Keep validation stable while training uses random SNR.
        val_snr = float(snr_values[0])

    return train_snr, val_snr, phase_name


def resolve_epoch_cbr_weight(epoch: int, args):
    """
    Returns:
        cbr_weight: float
        phase_name: str for logging
    """
    base_weight = float(args.cbr_weight)
    warmup_epochs = int(max(args.cbr_warmup_epochs, 0))
    finetune_epochs = int(max(args.cbr_finetune_epochs, 0))

    warmup_weight = float(args.cbr_warmup_weight) if args.cbr_warmup_weight is not None else base_weight
    finetune_weight = float(args.cbr_finetune_weight) if args.cbr_finetune_weight is not None else base_weight

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return warmup_weight, "warmup"
    if finetune_epochs > 0 and epoch > args.epochs - finetune_epochs:
        return finetune_weight, "finetune"
    return base_weight, "main"


def is_metric_improved(metric_name: str, current: float, best: float, min_delta: float):
    if best is None:
        return True
    if metric_name == "psnr":
        return current > best + min_delta
    return current < best - min_delta


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
            if nonfinite_skip_count > args.max_nonfinite_batches:
                raise RuntimeError(
                    f"Too many non-finite batches in epoch {epoch}: "
                    f"{nonfinite_skip_count} > max_nonfinite_batches={args.max_nonfinite_batches}"
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
            if nonfinite_skip_count > args.max_nonfinite_batches:
                raise RuntimeError(
                    f"Too many non-finite batches in epoch {epoch}: "
                    f"{nonfinite_skip_count} > max_nonfinite_batches={args.max_nonfinite_batches}"
                )
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
        if (
            scheduler is not None
            and args.lr_step_unit == "batch"
            and optimizer_step % args.lr_step_interval == 0
        ):
            scheduler.step()

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


def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_resume_path(resume_path, output_dir=None):
    candidates = []
    if resume_path:
        candidates.append(resume_path)

    if resume_path and not os.path.isabs(resume_path):
        candidates.append(os.path.join(os.getcwd(), resume_path))

        if output_dir:
            candidates.append(os.path.join(output_dir, resume_path))
            candidates.append(os.path.join(os.path.abspath(output_dir), resume_path))

    unique_candidates = []
    seen = set()
    for path in candidates:
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            unique_candidates.append(path)

    for path in unique_candidates:
        if os.path.isfile(path):
            return path

    # Convenience fallback: when only a filename is given (e.g. "best.pt"),
    # search recursively in ./runs and choose the newest checkpoint.
    basename_only = resume_path and os.path.basename(resume_path) == resume_path
    if basename_only:
        runs_root = os.path.join(os.getcwd(), "runs")
        pattern = os.path.join(runs_root, "**", resume_path)
        matches = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
        if matches:
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            selected = matches[0]
            if len(matches) > 1:
                print(
                    "[Warn] Multiple resume checkpoints found under ./runs. "
                    f"Auto-selected newest: {selected}"
                )
                for idx, path in enumerate(matches[:8], start=1):
                    print(f"[Warn] candidate[{idx}] {path}")
            else:
                print(f"[Info] Resolved resume checkpoint from ./runs: {selected}")
            return selected

    tried = "\n".join(f"  - {p}" for p in unique_candidates)
    raise FileNotFoundError(
        "Resume checkpoint not found. Tried:\n"
        f"{tried}"
    )


def load_resume_state(
    resume_path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    output_dir=None,
    load_optimizer=True,
    strict=True,
    expected_lr_step_unit=None,
    expected_lr_step_interval=1,
):
    resume_path = resolve_resume_path(resume_path, output_dir=output_dir)

    print(f"[Info] Loading resume checkpoint: {resume_path}")
    checkpoint = _safe_torch_load(resume_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_state = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        model_state = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint format in: {resume_path}")

    try:
        incompatible = model.load_state_dict(model_state, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load model state from resume checkpoint. "
            "Try using --resume-ignore-mismatch if architecture changed."
        ) from exc

    if not strict:
        if incompatible.missing_keys:
            print(f"[Warn] Missing keys during resume: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"[Warn] Unexpected keys during resume: {len(incompatible.unexpected_keys)}")

    start_epoch = 1
    global_step = 0
    optimizer_step = 0
    best_val_loss = float("inf")
    best_val_psnr = float("-inf")
    early_stop_best_metric = None
    early_stop_bad_epochs = 0

    if not (load_optimizer and isinstance(checkpoint, dict) and "optimizer" in checkpoint):
        print("[Info] Resume mode: weights only (optimizer/scheduler reset).")
        return (
            start_epoch,
            global_step,
            optimizer_step,
            best_val_loss,
            best_val_psnr,
            early_stop_best_metric,
            early_stop_bad_epochs,
        )

    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except Exception as exc:
        print(f"[Warn] Failed to load optimizer state, using fresh optimizer. reason={exc}")

    resume_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    resume_lr_step_unit = resume_args.get("lr_step_unit", None)
    resume_lr_step_interval = resume_args.get("lr_step_interval", None)
    skip_scheduler_load = False

    if expected_lr_step_unit is not None:
        if resume_lr_step_unit is None:
            if str(expected_lr_step_unit) == "batch":
                skip_scheduler_load = True
                print(
                    "[Warn] Resume checkpoint does not contain lr_step_unit metadata. "
                    "Current run uses batch-wise LR stepping, so scheduler state is reset."
                )
        else:
            resume_lr_step_unit = str(resume_lr_step_unit)
            resume_lr_step_interval = int(resume_lr_step_interval) if resume_lr_step_interval is not None else 1
            if (
                resume_lr_step_unit != str(expected_lr_step_unit)
                or resume_lr_step_interval != int(expected_lr_step_interval)
            ):
                skip_scheduler_load = True
                print(
                    "[Warn] Resume checkpoint LR stepping config differs from current run; "
                    "scheduler state is reset. "
                    f"resume=({resume_lr_step_unit}, {resume_lr_step_interval}) "
                    f"current=({expected_lr_step_unit}, {expected_lr_step_interval})"
                )

    if (not skip_scheduler_load) and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception as exc:
            print(f"[Warn] Failed to load scheduler state, using fresh scheduler. reason={exc}")

    if scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
        except Exception as exc:
            print(f"[Warn] Failed to load AMP scaler state, using fresh scaler. reason={exc}")

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    optimizer_step = int(checkpoint.get("optimizer_step", global_step))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    best_val_psnr = float(checkpoint.get("best_val_psnr", float("-inf")))
    if checkpoint.get("early_stop_best_metric", None) is not None:
        early_stop_best_metric = float(checkpoint.get("early_stop_best_metric"))
    early_stop_bad_epochs = int(checkpoint.get("early_stop_bad_epochs", 0))

    psnr_text = f"{best_val_psnr:.6f}" if math.isfinite(best_val_psnr) else "n/a"

    print(
        f"[Info] Resume success: start_epoch={start_epoch}, "
        f"global_step={global_step}, optimizer_step={optimizer_step}, best_val_loss={best_val_loss:.6f}, "
        f"best_val_psnr={psnr_text}, early_stop_bad_epochs={early_stop_bad_epochs}"
    )
    return (
        start_epoch,
        global_step,
        optimizer_step,
        best_val_loss,
        best_val_psnr,
        early_stop_best_metric,
        early_stop_bad_epochs,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    snr_values = parse_multiple_snr(args.multiple_snr)

    if args.val_interval_steps < 0:
        raise ValueError("--val-interval-steps must be >= 0.")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0.")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early-stop-min-delta must be >= 0.")
    if args.max_nonfinite_batches < 0:
        raise ValueError("--max-nonfinite-batches must be >= 0.")
    if args.lr_step_interval <= 0:
        raise ValueError("--lr-step-interval must be > 0.")
    if args.train_repeat <= 0:
        raise ValueError("--train-repeat must be > 0.")
    if args.val_repeat <= 0:
        raise ValueError("--val-repeat must be > 0.")
    if args.cbr_weight < 0:
        raise ValueError("--cbr-weight must be >= 0.")
    if args.cbr_bits_per_component <= 0:
        raise ValueError("--cbr-bits-per-component must be > 0.")
    if args.snr_warmup_epochs < 0:
        raise ValueError("--snr-warmup-epochs must be >= 0.")
    if args.snr_finetune_epochs < 0:
        raise ValueError("--snr-finetune-epochs must be >= 0.")
    if args.cbr_warmup_epochs < 0:
        raise ValueError("--cbr-warmup-epochs must be >= 0.")
    if args.cbr_finetune_epochs < 0:
        raise ValueError("--cbr-finetune-epochs must be >= 0.")
    if args.cbr_warmup_weight is not None and args.cbr_warmup_weight < 0:
        raise ValueError("--cbr-warmup-weight must be >= 0.")
    if args.cbr_finetune_weight is not None and args.cbr_finetune_weight < 0:
        raise ValueError("--cbr-finetune-weight must be >= 0.")
    if args.snr_warmup_epochs + args.snr_finetune_epochs > args.epochs:
        raise ValueError("--snr-warmup-epochs + --snr-finetune-epochs must be <= --epochs.")
    if args.cbr_warmup_epochs + args.cbr_finetune_epochs > args.epochs:
        raise ValueError("--cbr-warmup-epochs + --cbr-finetune-epochs must be <= --epochs.")

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
    config = make_config(args, device)
    model = MVSCNet(model_args, config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_step_unit == "batch":
        scheduler_steps_per_epoch = max(1, math.ceil(len(train_loader) / args.lr_step_interval))
        scheduler_t_max = max(1, args.epochs * scheduler_steps_per_epoch)
    else:
        scheduler_t_max = max(1, math.ceil(args.epochs / args.lr_step_interval))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max)

    scaler = make_grad_scaler(args.amp, device)

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_psnr = float("-inf")
    early_stop_best_metric = None
    early_stop_bad_epochs = 0
    global_step = 0
    optimizer_step = 0
    last_interval_val = None
    last_interval_step = -1
    current_val_snr = None

    if args.resume is not None:
        (
            start_epoch,
            global_step,
            optimizer_step,
            best_val_loss,
            best_val_psnr,
            early_stop_best_metric,
            early_stop_bad_epochs,
        ) = load_resume_state(
            resume_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            output_dir=args.output_dir,
            load_optimizer=not args.resume_weights_only,
            strict=not args.resume_ignore_mismatch,
            expected_lr_step_unit=args.lr_step_unit,
            expected_lr_step_interval=args.lr_step_interval,
        )

    if early_stop_best_metric is None:
        if args.early_stop_metric == "psnr" and math.isfinite(best_val_psnr):
            early_stop_best_metric = best_val_psnr
        elif args.early_stop_metric == "loss" and math.isfinite(best_val_loss):
            early_stop_best_metric = best_val_loss

    if start_epoch > args.epochs:
        print(
            f"[Info] Resume epoch {start_epoch} is already beyond target epochs {args.epochs}. "
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
            "early_stop_best_metric": early_stop_best_metric,
            "early_stop_bad_epochs": early_stop_bad_epochs,
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

    def run_interval_validation(global_step, epoch, train_step):
        nonlocal last_interval_val, last_interval_step, current_val_snr
        print(
            f"[Info] Interval validation at global_step={global_step} "
            f"(epoch={epoch}, train_step={train_step})"
        )
        val_loss, val_psnr, val_dist, val_cbr = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            args=args,
            epoch=epoch,
            given_snr_override=current_val_snr,
        )
        last_interval_val = (val_loss, val_psnr, val_dist, val_cbr)
        last_interval_step = global_step
        maybe_save_best(val_loss, epoch, global_step, source=f"interval@{global_step}")
        maybe_save_best_psnr(val_psnr, epoch, global_step, source=f"interval@{global_step}")

    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(
        f"[Info] LR scheduler: cosine, unit={args.lr_step_unit}, "
        f"interval={args.lr_step_interval}, T_max={scheduler_t_max}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_train_snr, epoch_val_snr, snr_phase = resolve_epoch_snr(epoch, args, snr_values)
        epoch_cbr_weight, cbr_phase = resolve_epoch_cbr_weight(epoch, args)
        model.cbr_weight = float(epoch_cbr_weight)
        current_val_snr = epoch_val_snr
        train_snr_desc = f"{epoch_train_snr:.3f}" if epoch_train_snr is not None else "random"
        print(
            f"[Info] epoch={epoch} snr_phase={snr_phase} "
            f"train_snr={train_snr_desc} val_snr={epoch_val_snr:.3f} "
            f"cbr_phase={cbr_phase} cbr_weight={model.cbr_weight:.3f}"
        )

        train_loss, train_psnr, train_dist, train_cbr, global_step, optimizer_step, train_nonfinite_skips = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
            given_snr_override=epoch_train_snr,
            global_step=global_step,
            optimizer_step=optimizer_step,
            val_interval_steps=args.val_interval_steps,
            on_val_interval=run_interval_validation,
        )

        if args.val_interval_steps > 0 and last_interval_step == global_step and last_interval_val is not None:
            val_loss, val_psnr, val_dist, val_cbr = last_interval_val
            print(
                f"[Info] Skip epoch-end validation at global_step={global_step} "
                "because interval validation already ran."
            )
        else:
            val_loss, val_psnr, val_dist, val_cbr = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                args=args,
                epoch=epoch,
                given_snr_override=epoch_val_snr,
            )

        maybe_save_best(val_loss, epoch, global_step, source="epoch-end")
        maybe_save_best_psnr(val_psnr, epoch, global_step, source="epoch-end")

        monitored_value = val_psnr if args.early_stop_metric == "psnr" else val_loss
        if is_metric_improved(
            metric_name=args.early_stop_metric,
            current=float(monitored_value),
            best=early_stop_best_metric,
            min_delta=float(args.early_stop_min_delta),
        ):
            early_stop_best_metric = float(monitored_value)
            early_stop_bad_epochs = 0
        else:
            early_stop_bad_epochs += 1
            print(
                f"[Info] EarlyStop wait: metric={args.early_stop_metric} "
                f"bad_epochs={early_stop_bad_epochs}/{args.early_stop_patience} "
                f"min_delta={args.early_stop_min_delta}"
            )

        if args.lr_step_unit == "epoch" and epoch % args.lr_step_interval == 0:
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

        if args.early_stop_patience > 0 and early_stop_bad_epochs >= args.early_stop_patience:
            print(
                f"[Info] Early stopping triggered at epoch={epoch}. "
                f"metric={args.early_stop_metric}, bad_epochs={early_stop_bad_epochs}."
            )
            break

    print("Training finished.")


if __name__ == "__main__":
    main()
