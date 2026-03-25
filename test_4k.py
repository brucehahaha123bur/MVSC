import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torch.utils.data import ConcatDataset, DataLoader

from data.harmony4d_mvsc import Harmony4DMVSCDataset

from net import encoder2 as enc_mod
from net import decoder2 as dec_mod

# NOTE:
# This trainer uses the updated encoder/decoder pipeline from `encoder2.py`
# and `decoder2.py`, together with the multi-scene Harmony4D dataset loader.


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
    parser = argparse.ArgumentParser(description="Evaluate MVSCNet on full-resolution Harmony4D GOP data")

    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_HARMONY_VAL_EXO_ROOT,
        help="Path to one exo folder, one scene folder, or a higher-level parent directory containing multiple Harmony4D scenes",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--output-dir", type=str, default="runs/mvsc_test_4k", help="Directory for logs/results")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Model construction crop size. The dataset itself runs in full-resolution mode with crop_size=None.",
    )
    parser.add_argument("--resize-shorter-to", type=int, default=0)
    parser.add_argument("--val-repeat", type=int, default=100)
    parser.add_argument("--min-common-frames", type=int, default=8)
    parser.add_argument(
        "--fixed-cam-names",
        type=str,
        default="",
        help="Comma-separated fixed camera names, for example: cam02,cam03,cam11,cam17",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional limit on the number of deterministic test samples to run. 0 means run all samples.",
    )

    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--individual-depths", type=str, default="1,2,1")
    parser.add_argument("--common-depths", type=str, default="2,1")
    parser.add_argument("--common-heads", type=str, default="8,10")

    parser.add_argument("--channel-type", type=str, default="awgn", choices=["awgn", "rayleigh", "none"])
    parser.add_argument("--multiple-snr", type=str, default="15", help="Comma-separated SNR list, e.g. 1,4,7,10")
    parser.add_argument("--test-snr", type=float, default=None, help="Optional fixed SNR used during evaluation")
    parser.add_argument("--cbr-weight", type=float, default=0.0)
    parser.add_argument(
        "--cbr-bits-per-component",
        type=float,
        default=3.0,
        help="Bit depth per transmitted IQ component used by CBR accounting (default: 3.0)",
    )
    parser.add_argument("--distortion-metric", type=str, default="MSE", choices=["MSE", "SSIM", "MS-SSIM"])

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable automatic mixed precision on CUDA")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision on CUDA")
    parser.set_defaults(amp=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-images", action="store_true", help="Save reconstruction images for a subset of batches")
    parser.add_argument("--max-save-batches", type=int, default=5)

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


# Helper for parsing fixed camera names
def parse_fixed_cam_names(spec: str):
    names = []
    for token in str(spec).split(","):
        token = token.strip()
        if token:
            names.append(token)
    return names if names else None


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

def build_dataset(root, args):
    resize_shorter_to = args.resize_shorter_to if args.resize_shorter_to > 0 else None
    root_list = parse_root_list(root)
    fixed_cam_names = parse_fixed_cam_names(args.fixed_cam_names)

    for r in root_list:
        if not os.path.isdir(r):
            raise FileNotFoundError(f"Dataset root does not exist: {r}")

    if len(root_list) == 1:
        root_abs = root_list[0]
        print(f"[Info] Test uses root: {root_abs}")
        return Harmony4DMVSCDataset(
            root=root_abs,
            num_views=args.num_views,
            num_frames=args.num_frames,
            crop_size=None,
            resize_shorter_to=resize_shorter_to,
            random_crop=False,
            random_flip=False,
            min_common_frames=args.min_common_frames,
            repeat=max(int(args.val_repeat), 1),
            fixed_cam_names=fixed_cam_names,
            deterministic=True,
        )

    print(f"[Info] Test uses {len(root_list)} roots:")
    datasets = []
    for i, r in enumerate(root_list, start=1):
        print(f"[Info]   [{i}] root={r}")
        datasets.append(
            Harmony4DMVSCDataset(
                root=r,
                num_views=args.num_views,
                num_frames=args.num_frames,
                crop_size=None,
                resize_shorter_to=resize_shorter_to,
                random_crop=False,
                random_flip=False,
                min_common_frames=args.min_common_frames,
                repeat=max(int(args.val_repeat), 1),
                fixed_cam_names=fixed_cam_names,
                deterministic=True,
            )
        )

    return ConcatDataset(datasets)


def build_loader(dataset, args, device):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
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


# Helper: convert a tensor to a uint8 HWC numpy array
def tensor_to_uint8_image(x: torch.Tensor):
    x = x.detach().cpu().clamp(0.0, 1.0)
    x = (x * 255.0).round().byte().permute(1, 2, 0).numpy()
    return x


def save_frame_tensor_as_image(x: torch.Tensor, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(x)).save(str(save_path))


def maybe_save_reconstructions(batch, x_hat, output_dir: Path, batch_idx: int):
    x = batch["x"]
    scene_names = _flatten_string_fields(batch.get("scene_name", []))
    cam_names = batch.get("cam_names", [])
    frame_names = batch.get("frame_names", [])

    if x.ndim != 6:
        return

    batch_size = x.shape[0]
    for b in range(batch_size):
        scene_name = scene_names[b] if b < len(scene_names) else f"sample_{batch_idx:04d}_{b:02d}"
        scene_dir = output_dir / f"batch_{batch_idx:04d}" / scene_name

        cams_b = cam_names[b] if isinstance(cam_names, (list, tuple)) and b < len(cam_names) else cam_names
        frames_b = frame_names[b] if isinstance(frame_names, (list, tuple)) and b < len(frame_names) else frame_names
        cams_b = _flatten_string_fields(cams_b)
        frames_b = _flatten_string_fields(frames_b)

        for t in range(x.shape[1]):
            frame_tag = frames_b[t] if t < len(frames_b) else f"t{t:02d}"
            for v in range(x.shape[2]):
                cam_tag = cams_b[v] if v < len(cams_b) else f"v{v:02d}"
                gt_path = scene_dir / f"gt_{cam_tag}_{frame_tag}.png"
                rec_path = scene_dir / f"rec_{cam_tag}_{frame_tag}.png"
                save_frame_tensor_as_image(x[b, t, v], gt_path)
                save_frame_tensor_as_image(x_hat[b, t, v], rec_path)


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
        self.train_stage = "full"
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

        # IMPORTANT: individual encoder outputs 192 channels, so common encoder must take 192
        self.common_encoder = enc_mod.MVSC_Commonality_Encoder(
            dim=192,
            input_resolution=(self.crop_size // 8, self.crop_size // 8),
            depths=self.common_depths,
            num_heads=self.common_heads,
        )

        # IMPORTANT: common encoder outputs 320 channels, so JSCC encoder must take 320
        self.jscc_encoder = enc_mod.MVSC_JSCC_Encoder(
            dim=320,
            latent_dim=self.latent_dim,
        )

        # Current commonality encoder compresses both time T and views V once.
        # Therefore JSCC decoder must restore with compressed view count Vc = num_views // 2.
        self.jscc_decoder = dec_mod.MVSC_JSCC_Decoder(
            latent_dim=self.latent_dim,
            embed_dim=320,
            compressed_num_views=self.num_views // 2,
            temporal_upsample_in_jscc=False,
        )

        # IMPORTANT: decoder stage order is reversed relative to encoder:
        # encoder heads/depths are (stage1=256, stage2=320) -> (8, 10) and (2, 1).
        # The commonality decoder now also needs the compressed view count because
        # encoder2 merges the view axis once before JSCC.
        self.common_decoder = dec_mod.MVSC_Commonality_Decoder(
            dim=320,
            input_resolution=(self.crop_size // 8, self.crop_size // 8),
            num_views=self.num_views,
            compressed_num_views=self.num_views // 2,
            depths=tuple(reversed(self.common_depths)),
            num_heads=tuple(reversed(self.common_heads)),
        )

        # IMPORTANT: individual encoder outputs C=192, not token_dim
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

    @staticmethod
    def _set_resolution_recursive(module: nn.Module, resolution):
        if module is None:
            return
        if hasattr(module, "input_resolution"):
            module.input_resolution = tuple(int(v) for v in resolution)
        if hasattr(module, "img_size"):
            img_size = getattr(module, "img_size")
            if isinstance(img_size, (tuple, list)) and len(img_size) == 2:
                module.img_size = tuple(int(v) for v in resolution)
        if hasattr(module, "attn_mask"):
            module.attn_mask = None
        if hasattr(module, "attn_masks"):
            module.attn_masks = None
        for child in module.children():
            ModernMVSCNet._set_resolution_recursive(child, resolution)

    def _configure_runtime_shapes(self, x: torch.Tensor):
        if x.ndim != 6:
            return

        _, _, _, _, H, W = x.shape
        if H % 8 != 0 or W % 8 != 0:
            raise ValueError(
                f"Full-resolution test currently expects H and W to be divisible by 8, got {(H, W)}"
            )

        full_res = (int(H), int(W))
        # individual_encoder uses patch_size=2, then two spatial downsampling stages.
        # Therefore the three Swin stages run at H/2, H/4, and H/8 respectively,
        # and the common encoder/decoder bottleneck also lives at H/8.
        stage_res_0 = (int(H // 2), int(W // 2))
        stage_res_1 = (int(H // 4), int(W // 4))
        bottleneck_res = (int(H // 8), int(W // 8))

        if hasattr(self, "individual_encoder") and self.individual_encoder is not None:
            if hasattr(self.individual_encoder, "img_size"):
                self.individual_encoder.img_size = full_res
            swin_layers = getattr(self.individual_encoder, "swin_layers", None)
            if isinstance(swin_layers, (list, tuple, nn.ModuleList)) and len(swin_layers) >= 3:
                self._set_resolution_recursive(swin_layers[0], stage_res_0)
                self._set_resolution_recursive(swin_layers[1], stage_res_1)
                self._set_resolution_recursive(swin_layers[2], bottleneck_res)

        if hasattr(self, "common_encoder") and self.common_encoder is not None:
            if hasattr(self.common_encoder, "input_resolution"):
                self.common_encoder.input_resolution = bottleneck_res
            self._set_resolution_recursive(self.common_encoder, bottleneck_res)

        if hasattr(self, "common_decoder") and self.common_decoder is not None:
            if hasattr(self.common_decoder, "input_resolution"):
                self.common_decoder.input_resolution = bottleneck_res
            self._set_resolution_recursive(self.common_decoder, bottleneck_res)

        if hasattr(self, "individual_decoder") and self.individual_decoder is not None:
            if hasattr(self.individual_decoder, "img_size"):
                self.individual_decoder.img_size = full_res
            if hasattr(self.individual_decoder, "input_resolution"):
                self.individual_decoder.input_resolution = bottleneck_res

            decoder_swin_layers = getattr(self.individual_decoder, "swin_layers", None)
            if isinstance(decoder_swin_layers, (list, tuple, nn.ModuleList)) and len(decoder_swin_layers) >= 3:
                self._set_resolution_recursive(decoder_swin_layers[0], bottleneck_res)
                self._set_resolution_recursive(decoder_swin_layers[1], stage_res_1)
                self._set_resolution_recursive(decoder_swin_layers[2], stage_res_0)

            decoder_layers = getattr(self.individual_decoder, "layers", None)
            if isinstance(decoder_layers, (list, tuple, nn.ModuleList)) and len(decoder_layers) >= 3:
                self._set_resolution_recursive(decoder_layers[0], bottleneck_res)
                self._set_resolution_recursive(decoder_layers[1], stage_res_1)
                self._set_resolution_recursive(decoder_layers[2], stage_res_0)

    @staticmethod
    def _pad_hw_to_multiple(x: torch.Tensor, multiple: int):
        if x.ndim != 6:
            return x, (0, 0), (0, 0)

        H, W = int(x.shape[-2]), int(x.shape[-1])
        target_h = ((H + multiple - 1) // multiple) * multiple
        target_w = ((W + multiple - 1) // multiple) * multiple
        pad_h = target_h - H
        pad_w = target_w - W

        if pad_h == 0 and pad_w == 0:
            return x, (H, W), (0, 0)

        b, t, v, c, _, _ = x.shape
        x_reshaped = x.view(b * t * v, c, H, W)
        x_reshaped = F.pad(x_reshaped, (0, pad_w, 0, pad_h), mode="replicate")
        x_padded = x_reshaped.view(b, t, v, c, target_h, target_w)
        return x_padded, (H, W), (pad_h, pad_w)

    @staticmethod
    def _crop_hw(x: torch.Tensor, hw):
        if x.ndim != 6:
            return x
        H, W = int(hw[0]), int(hw[1])
        return x[..., :H, :W]

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
        x_ref = x
        x_in, orig_hw, _ = self._pad_hw_to_multiple(x, multiple=16)

        self._configure_runtime_shapes(x_in)
        x_ind = self.individual_encoder(x_in)

        if self.train_stage == "individual_only":
            x_hat = self.individual_decoder(x_ind)
            x_hat = self._crop_hw(x_hat, orig_hw)
            used_snr = None
            distortion = torch.mean((x_hat - x_ref) ** 2)
            cbr = x_ref.new_tensor(0.0)
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
        cbr = x_ref.new_tensor(0.0)

        if self.train_stage == "full":
            y = self.jscc_encoder(x_com)
            used_snr = self._sample_snr(given_SNR)
            y_hat = self._apply_channel(y, used_snr)
            x_com_dec = self.jscc_decoder(y_hat)
            cbr = self._compute_cbr(y, x_ref)
        elif self.train_stage != "ic_only":
            raise ValueError(f"Unsupported train_stage: {self.train_stage}")

        x_com_dec = self.common_decoder(x_com_dec)
        x_hat = self.individual_decoder(x_com_dec)
        x_hat = self._crop_hw(x_hat, orig_hw)

        distortion = torch.mean((x_hat - x_ref) ** 2)
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
            # Do not call scaler.update() here: no backward/step was recorded for this skipped batch,
            # and torch.amp.GradScaler will assert if update() is called without prior inf checks.
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

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
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
    with open(os.path.join(args.output_dir, "test_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    test_dataset = build_dataset(args.data_root, args)
    if args.max_samples > 0 and len(test_dataset) > args.max_samples:
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, list(range(args.max_samples)))
        print(f"[Info] Restrict test samples to first {args.max_samples}")

    test_loader = build_loader(test_dataset, args, device=device)

    model_args = make_model_args(args)
    model = ModernMVSCNet(model_args).to(device)
    model.channel_type = args.channel_type

    print(f"[Info] Load checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint format: {args.checkpoint}")

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(f"[Info] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
    if missing:
        for key in missing[:10]:
            print(f"[Warn] missing: {key}")
        if len(missing) > 10:
            print(f"[Warn] ... and {len(missing) - 10} more missing keys")
    if unexpected:
        for key in unexpected[:10]:
            print(f"[Warn] unexpected: {key}")
        if len(unexpected) > 10:
            print(f"[Warn] ... and {len(unexpected) - 10} more unexpected keys")

    model.eval()

    print(
        f"[Info] Model config: embed_dim={args.embed_dim}, token_dim={4 * args.embed_dim}, latent_dim={args.latent_dim}, "
        f"individual_depths={args.individual_depths}, common_depths={args.common_depths}, common_heads={args.common_heads}, "
        f"compressed_num_views={args.num_views // 2}"
    )
    print(f"[Info] Test batches: {len(test_loader)}")
    print(
        "[Warn] This script feeds the dataset in full-resolution mode (crop_size=None), "
        "but ModernMVSCNet is still constructed with fixed resolution arguments. "
        "If encoder2/decoder2 are not fully dynamic, true 4K inference may still fail inside the model."
    )

    loss_meter = AverageMeter()
    distortion_meter = AverageMeter()
    cbr_meter = AverageMeter()
    psnr_meter = AverageMeter()
    image_output_dir = Path(args.output_dir) / "reconstructions"

    with torch.no_grad():
        for step, batch in enumerate(test_loader, start=1):
            x = batch["x"].to(device, non_blocking=True)

            if not torch.isfinite(x).all():
                print(f"[Warn] Non-finite input at test step={step}. {_batch_debug_desc(batch)}")
                continue

            if step == 1:
                print(f"[Info] First batch input shape: {tuple(x.shape)}")

            with autocast_context(args.amp, device):
                x_hat, used_snr, loss, aux = model(x, given_SNR=args.test_snr)

            if not (bool(torch.isfinite(loss).all().item()) and bool(torch.isfinite(x_hat).all().item())):
                print(
                    f"[Warn] Non-finite forward at test step={step} loss={float(loss.detach().float().mean().item()):.6f} "
                    f"snr={used_snr} {_batch_debug_desc(batch)}"
                )
                continue

            distortion_value = float(aux["distortion"].item()) if "distortion" in aux else float(loss.item())
            cbr_value = float(aux["cbr"].item()) if "cbr" in aux else 0.0
            psnr_value = compute_psnr(x_hat.detach(), x.detach())

            bsz = x.shape[0]
            loss_meter.update(loss.item(), bsz)
            distortion_meter.update(distortion_value, bsz)
            cbr_meter.update(cbr_value, bsz)
            psnr_meter.update(psnr_value, bsz)

            if args.save_images and step <= args.max_save_batches:
                maybe_save_reconstructions(batch, x_hat.detach().cpu(), image_output_dir, step)

            if step % args.log_interval == 0 or step == len(test_loader):
                print(
                    f"[Test] step={step}/{len(test_loader)} "
                    f"loss={loss_meter.val:.6f} avg_loss={loss_meter.avg:.6f} "
                    f"dist={distortion_meter.val:.6f} avg_dist={distortion_meter.avg:.6f} "
                    f"cbr={cbr_meter.val:.6f} avg_cbr={cbr_meter.avg:.6f} "
                    f"psnr={psnr_meter.val:.3f} avg_psnr={psnr_meter.avg:.3f} "
                    f"snr={used_snr}"
                )

    if loss_meter.count == 0:
        raise RuntimeError("Test produced no finite batches.")

    summary = {
        "avg_loss": loss_meter.avg,
        "avg_distortion": distortion_meter.avg,
        "avg_cbr": cbr_meter.avg,
        "avg_psnr": psnr_meter.avg,
        "num_samples": loss_meter.count,
        "checkpoint": args.checkpoint,
        "test_snr": args.test_snr,
        "channel_type": args.channel_type,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
