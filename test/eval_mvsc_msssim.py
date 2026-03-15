'''
python.exe "F:\新桌面\MVSC-main\test\eval_mvsc_msssim.py" --checkpoint "F:\新桌面\MVSC-main\runs\mvsc_detail_ft_cbr1024_sched\best_psnr.pt" --val-root "E:\Harmony4D\test\01_hugging\002_hugging\exo" --snr 15 --val-repeat 1 --output-dir "F:\新桌面\MVSC-main\runs\mvsc_msssim_eval_4k_snr15_full_tiled_smoke" --device cuda --full-image --full-image-side max --tile-size 512 --tile-stride 512
'''
import argparse
import csv
import glob
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from net.network import MVSCNet
from train_mvsc import (
    AverageMeter,
    build_dataset,
    build_loader,
    make_config,
    make_model_args,
    evaluate,
    resolve_device,
    resolve_resume_path,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MVSC checkpoint with MS-SSIM distortion on validation set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path, e.g. ./runs/exp/best_psnr.pt")
    parser.add_argument("--train-args", type=str, default=None, help="Path to train_args.json (default: sibling of checkpoint)")
    parser.add_argument("--val-root", type=str, default=None, help="Override validation root")
    parser.add_argument("--output-dir", type=str, default="./runs/mvsc_msssim_eval", help="Directory for evaluation CSV")
    parser.add_argument("--snr", type=float, default=15.0, help="Fixed SNR for evaluation")
    parser.add_argument("--val-repeat", type=int, default=None, help="Override val_repeat for this evaluation")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 recommended)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Runtime device")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic validation sampling")
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Use full-resolution frame as model input by expanding crop_size to image side (short side is zero-padded)",
    )
    parser.add_argument(
        "--full-image-side",
        type=str,
        default="max",
        choices=["max", "width", "height"],
        help="Target square side when --full-image is enabled (max is recommended for full coverage)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile side for full-image evaluation; should be divisible by 128 for MVSC",
    )
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=512,
        help="Tile stride for full-image evaluation; <= tile-size gives overlapping blending",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP in tiled full-image inference (AMP is enabled by default to reduce memory)",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_train_args(args, checkpoint_path: str):
    if args.train_args is not None:
        train_args_path = args.train_args
    else:
        train_args_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "train_args.json")

    if not os.path.isfile(train_args_path):
        raise FileNotFoundError(
            "train_args.json not found. Please pass --train-args explicitly. "
            f"Tried: {train_args_path}"
        )

    with open(train_args_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    train_args = SimpleNamespace(**raw)

    # Backward compatibility for older runs.
    if not hasattr(train_args, "repeat_per_exo"):
        train_args.repeat_per_exo = False
    if not hasattr(train_args, "max_nonfinite_batches"):
        train_args.max_nonfinite_batches = 20
    if not hasattr(train_args, "cbr_weight"):
        train_args.cbr_weight = 8192.0
    if not hasattr(train_args, "cbr_bits_per_component"):
        train_args.cbr_bits_per_component = 3.0

    return train_args, train_args_path


def load_checkpoint_ignoring_distortion(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    # Distortion module buffers depend on distortion_metric; ignore them for cross-metric evaluation.
    # Full-image mode also changes Swin attention mask buffer shapes, so we skip those buffers as well.
    model_state = model.state_dict()
    param_keys = set(dict(model.named_parameters()).keys())

    filtered_state = {}
    skipped_buffer_keys = set()
    skipped_unexpected_keys = set()
    shape_mismatch_param_keys = []

    for key, value in state_dict.items():
        if key.startswith("distortion_loss."):
            skipped_buffer_keys.add(key)
            continue

        if key.endswith("attn_mask"):
            skipped_buffer_keys.add(key)
            continue

        if key not in model_state:
            skipped_unexpected_keys.add(key)
            continue

        target = model_state[key]
        if tuple(value.shape) != tuple(target.shape):
            # Only allow shape mismatch on buffers (e.g., resolution-dependent masks).
            if key in param_keys:
                shape_mismatch_param_keys.append((key, tuple(value.shape), tuple(target.shape)))
            else:
                skipped_buffer_keys.add(key)
            continue

        filtered_state[key] = value

    if shape_mismatch_param_keys:
        preview = ", ".join(
            [f"{k}: ckpt{src} != model{dst}" for k, src, dst in shape_mismatch_param_keys[:8]]
        )
        raise RuntimeError(
            "Checkpoint has parameter shape mismatches (likely architecture mismatch), "
            f"cannot continue. details={preview}"
        )

    incompatible = model.load_state_dict(filtered_state, strict=False)

    missing_non_ignored = [
        k
        for k in incompatible.missing_keys
        if (k not in skipped_buffer_keys) and (not k.startswith("distortion_loss."))
    ]
    unexpected_non_ignored = [
        k
        for k in incompatible.unexpected_keys
        if (k not in skipped_unexpected_keys) and (not k.startswith("distortion_loss."))
    ]

    if missing_non_ignored or unexpected_non_ignored:
        raise RuntimeError(
            "Checkpoint is incompatible with current model definition. "
            f"missing_non_ignored={missing_non_ignored[:8]}, unexpected_non_ignored={unexpected_non_ignored[:8]}"
        )


def _find_first_image(root_dir: str):
    root_dir = os.path.abspath(root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Validation root does not exist: {root_dir}")

    images_dirs = []
    for cur_root, _, _ in os.walk(root_dir):
        if os.path.basename(cur_root).lower() == "images":
            images_dirs.append(cur_root)

    for images_dir in sorted(images_dirs):
        for name in sorted(os.listdir(images_dir)):
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            return os.path.join(images_dir, name)

    # Fallback: any image file under root.
    for cur_root, _, files in os.walk(root_dir):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTS:
                return os.path.join(cur_root, name)

    raise RuntimeError(f"No image files found under validation root: {root_dir}")


def _resolve_full_image_crop_size(val_root: str, side_mode: str):
    probe_path = _find_first_image(val_root)
    with Image.open(probe_path) as img:
        width, height = img.size

    if side_mode == "max":
        crop_size = int(max(width, height))
    elif side_mode == "width":
        crop_size = int(width)
    elif side_mode == "height":
        crop_size = int(height)
    else:
        raise ValueError(f"Unsupported full-image-side: {side_mode}")

    if crop_size <= 0:
        raise ValueError(f"Invalid crop_size from image size: width={width}, height={height}")

    return probe_path, width, height, crop_size


def _warn_if_checkpoint_is_filename_only(checkpoint_arg: str):
    normalized = str(checkpoint_arg).replace("\\", "/").strip()
    filename_only = ("/" not in normalized) and (not os.path.isabs(normalized))
    if filename_only and (not os.path.isfile(checkpoint_arg)):
        print(
            "[Warn] --checkpoint looks like a filename-only argument. "
            "This repo has multiple runs with same checkpoint names, so auto-resolve may pick a different run. "
            "Prefer an explicit path like runs/mvsc_detail_ft_cbr1024_sched/best_psnr.pt"
        )


def _resolve_checkpoint_path(checkpoint_arg: str):
    normalized = str(checkpoint_arg).replace("\\", "/").strip()
    filename_only = ("/" not in normalized) and (not os.path.isabs(normalized))

    if filename_only and (not os.path.isfile(checkpoint_arg)):
        runs_root = os.path.join(ROOT_DIR, "runs")
        pattern = os.path.join(runs_root, "**", checkpoint_arg)
        matches = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
        if len(matches) > 1:
            matches = sorted(matches)
            preview = "\n".join([f"  - {m}" for m in matches[:8]])
            raise ValueError(
                "Ambiguous checkpoint filename. Multiple candidates found; pass an explicit path via --checkpoint.\n"
                f"Candidates:\n{preview}"
            )

    return resolve_resume_path(checkpoint_arg, output_dir=os.path.join(ROOT_DIR, "runs"))


def _make_starts(total: int, tile: int, stride: int):
    if tile >= total:
        return [0]

    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def _autocast_context(use_amp: bool, device):
    if not use_amp or device.type != "cuda":
        return nullcontext()

    # Prefer modern API, with safe fallback for older torch versions.
    try:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def _effective_tile_size(requested: int, h: int, w: int):
    if requested <= 0:
        raise ValueError("--tile-size must be > 0")

    tile = min(int(requested), int(h), int(w))
    tile = (tile // 128) * 128
    if tile < 128:
        raise ValueError(
            f"Effective tile size is too small for MVSC constraints: requested={requested}, H={h}, W={w}"
        )
    return int(tile)


def _compute_distortion_sequential(model, x_ref_cpu, x_hat_cpu, device):
    # Keep GPU memory low by evaluating MS-SSIM one image at a time.
    b, t, v, c, h, w = x_ref_cpu.shape
    x_ref_flat = x_ref_cpu.reshape(b * t * v, c, h, w)
    x_hat_flat = x_hat_cpu.reshape(b * t * v, c, h, w)

    values = []
    for idx in range(x_ref_flat.shape[0]):
        ref_i = x_ref_flat[idx : idx + 1].to(device, non_blocking=True)
        hat_i = x_hat_flat[idx : idx + 1].to(device, non_blocking=True)
        dist_i = model.distortion_loss.forward(ref_i, hat_i, normalization=model.config.norm)
        values.append(float(dist_i.item()))
        del ref_i
        del hat_i
        del dist_i
    return float(sum(values) / max(len(values), 1))


def evaluate_full_image_tiled(model, loader, device, args, epoch, given_snr_override, tile_size, tile_stride, use_amp):
    model.eval()
    loss_meter = AverageMeter()
    distortion_meter = AverageMeter()
    cbr_meter = AverageMeter()
    psnr_meter = AverageMeter()
    nonfinite_skip_count = 0

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            x_cpu = batch["x"].float()

            if not torch.isfinite(x_cpu).all():
                nonfinite_skip_count += 1
                print(
                    f"[Warn] Non-finite input in tiled validation at epoch={epoch} step={step}. "
                    f"skip_count={nonfinite_skip_count}"
                )
                continue

            bsz, _, _, _, h, w = x_cpu.shape
            eff_tile = _effective_tile_size(tile_size, h, w)
            eff_stride = int(tile_stride) if int(tile_stride) > 0 else eff_tile
            if eff_stride > eff_tile:
                eff_stride = eff_tile

            h_starts = _make_starts(h, eff_tile, eff_stride)
            w_starts = _make_starts(w, eff_tile, eff_stride)

            x_hat_acc = torch.zeros_like(x_cpu)
            weight_acc = torch.zeros((bsz, 1, 1, 1, h, w), dtype=x_cpu.dtype)

            transmitted_bits_sum = 0.0
            source_values_sum = 0.0
            used_snr = float(given_snr_override)

            for top in h_starts:
                for left in w_starts:
                    bottom = top + eff_tile
                    right = left + eff_tile

                    x_tile = x_cpu[..., top:bottom, left:right].to(device, non_blocking=True)

                    with _autocast_context(use_amp=use_amp, device=device):
                        model_out = model(x_tile, given_SNR=given_snr_override)

                    if isinstance(model_out, tuple) and len(model_out) == 4:
                        x_hat_tile, used_snr, _, aux = model_out
                        cbr_tile = float(aux["cbr"].item()) if "cbr" in aux else 0.0
                    else:
                        x_hat_tile, used_snr, _ = model_out
                        cbr_tile = 0.0

                    source_values = float(x_tile.numel())
                    transmitted_bits_sum += cbr_tile * source_values
                    source_values_sum += source_values

                    x_hat_cpu = x_hat_tile.detach().float().cpu()
                    x_hat_acc[..., top:bottom, left:right] += x_hat_cpu
                    weight_acc[..., top:bottom, left:right] += 1.0

                    del x_tile
                    del x_hat_tile
                    del x_hat_cpu

            x_hat_full = x_hat_acc / weight_acc.clamp_min(1.0)
            cbr_value = transmitted_bits_sum / max(source_values_sum, 1.0)

            mse = torch.mean((x_hat_full - x_cpu) ** 2).item()
            mse = max(mse, 1e-12)
            psnr_value = -10.0 * math.log10(mse)

            distortion_value = _compute_distortion_sequential(model, x_cpu, x_hat_full, device)
            loss_value = distortion_value + float(getattr(model, "cbr_weight", 0.0)) * cbr_value

            loss_meter.update(loss_value, bsz)
            distortion_meter.update(distortion_value, bsz)
            cbr_meter.update(cbr_value, bsz)
            psnr_meter.update(psnr_value, bsz)

            if step % args.log_interval == 0 or step == len(loader):
                print(
                    f"[ValTiled] epoch={epoch} step={step}/{len(loader)} "
                    f"loss={loss_meter.val:.6f} avg_loss={loss_meter.avg:.6f} "
                    f"dist={distortion_meter.val:.6f} avg_dist={distortion_meter.avg:.6f} "
                    f"cbr={cbr_meter.val:.6f} avg_cbr={cbr_meter.avg:.6f} "
                    f"psnr={psnr_meter.val:.3f} avg_psnr={psnr_meter.avg:.3f} "
                    f"snr={used_snr} tile={eff_tile} stride={eff_stride} amp={use_amp}"
                )

            # Release GPU cache periodically in low-memory scenarios.
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if loss_meter.count == 0:
        raise RuntimeError(
            f"Tiled validation produced no finite batches at epoch={epoch}. "
            f"skipped={nonfinite_skip_count}"
        )

    if nonfinite_skip_count > 0:
        print(f"[Warn] Epoch {epoch} skipped non-finite tiled val batches: {nonfinite_skip_count}")

    return loss_meter.avg, psnr_meter.avg, distortion_meter.avg, cbr_meter.avg


def main():
    args = parse_args()
    set_seed(args.seed)

    _warn_if_checkpoint_is_filename_only(args.checkpoint)
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    train_args, train_args_path = load_train_args(args, checkpoint_path)

    train_args.distortion_metric = "MS-SSIM"
    train_args.device = args.device
    train_args.num_workers = int(args.num_workers)
    train_args.log_interval = int(1e9)
    if args.val_root is not None:
        train_args.val_root = args.val_root
    if args.val_repeat is not None:
        train_args.val_repeat = int(args.val_repeat)

    val_root = train_args.val_root if train_args.val_root is not None else train_args.train_root
    use_tiled_eval = bool(args.full_image)

    # model_args_for_build controls network shape assumptions at construction time.
    # data loading may intentionally use a different crop_size in full-image mode.
    model_args_for_build = deepcopy(train_args)

    full_image_info = None
    if args.full_image:
        probe_path, img_w, img_h, full_crop_size = _resolve_full_image_crop_size(val_root, args.full_image_side)
        train_args.crop_size = int(full_crop_size)
        # Disable resize stage so the loader keeps original pixel dimensions.
        train_args.resize_shorter_to = -1
        full_image_info = {
            "probe_path": probe_path,
            "image_width": int(img_w),
            "image_height": int(img_h),
            "crop_size": int(full_crop_size),
            "side_mode": args.full_image_side,
        }

        # Tiled inference runs model on tiles, so model init resolution must match tile size,
        # not full-image crop size.
        runtime_tile = _effective_tile_size(int(args.tile_size), int(full_crop_size), int(full_crop_size))
        model_args_for_build.crop_size = int(runtime_tile)

    device = resolve_device(train_args.device)
    if device.type != "cuda":
        raise ValueError("MS-SSIM evaluation in this codebase requires CUDA. Use --device cuda.")

    model = MVSCNet(make_model_args(model_args_for_build), make_config(model_args_for_build, device)).to(device)
    load_checkpoint_ignoring_distortion(model, checkpoint_path, device)
    model.eval()

    val_dataset = build_dataset(val_root, train_args, is_train=False)
    val_loader = build_loader(val_dataset, train_args, is_train=False, device=device)

    amp_enabled = (not args.no_amp)
    if use_tiled_eval:
        val_loss, val_psnr, val_dist, val_cbr = evaluate_full_image_tiled(
            model=model,
            loader=val_loader,
            device=device,
            args=train_args,
            epoch=0,
            given_snr_override=float(args.snr),
            tile_size=int(args.tile_size),
            tile_stride=int(args.tile_stride),
            use_amp=amp_enabled,
        )
    else:
        val_loss, val_psnr, val_dist, val_cbr = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            args=train_args,
            epoch=0,
            given_snr_override=float(args.snr),
        )

    # In this project, MS-SSIM distortion is defined as (1 - MS-SSIM).
    val_msssim = 1.0 - float(val_dist)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "msssim_eval.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snr",
                "val_psnr",
                "val_msssim",
                "val_dist_1_minus_msssim",
                "val_loss",
                "val_cbr",
                "val_repeat",
                "checkpoint",
                "train_args",
                "val_root",
                "cbr_bits_per_component",
                "crop_size",
                "full_image",
                "full_image_side",
                "full_image_probe",
                "full_image_width",
                "full_image_height",
                "tiled_eval",
                "tile_size",
                "tile_stride",
                "amp_enabled",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "snr": float(args.snr),
                "val_psnr": float(val_psnr),
                "val_msssim": float(val_msssim),
                "val_dist_1_minus_msssim": float(val_dist),
                "val_loss": float(val_loss),
                "val_cbr": float(val_cbr),
                "val_repeat": int(train_args.val_repeat),
                "checkpoint": checkpoint_path,
                "train_args": train_args_path,
                "val_root": val_root,
                "cbr_bits_per_component": float(getattr(train_args, "cbr_bits_per_component", 3.0)),
                "crop_size": int(train_args.crop_size),
                "full_image": bool(args.full_image),
                "full_image_side": str(args.full_image_side),
                "full_image_probe": "" if full_image_info is None else str(full_image_info["probe_path"]),
                "full_image_width": "" if full_image_info is None else int(full_image_info["image_width"]),
                "full_image_height": "" if full_image_info is None else int(full_image_info["image_height"]),
                "tiled_eval": bool(use_tiled_eval),
                "tile_size": int(args.tile_size),
                "tile_stride": int(args.tile_stride),
                "amp_enabled": bool(amp_enabled),
            }
        )

    print(f"checkpoint={checkpoint_path}")
    print(f"train_args={train_args_path}")
    print(f"val_root={val_root}")
    print(f"crop_size={int(train_args.crop_size)}")
    print(f"distortion_metric={train_args.distortion_metric}")
    print(f"tiled_eval={bool(use_tiled_eval)}")
    print(f"tile_size={int(args.tile_size)}")
    print(f"tile_stride={int(args.tile_stride)}")
    print(f"amp_enabled={bool(amp_enabled)}")
    if full_image_info is not None:
        print(
            "full_image_mode=on "
            f"side={full_image_info['side_mode']} "
            f"source={full_image_info['image_width']}x{full_image_info['image_height']} "
            f"probe={full_image_info['probe_path']}"
        )
        print("note=non-square images are zero-padded to square for MVSC shape constraints")
    print(f"snr={float(args.snr):.2f}")
    print(f"val_psnr={float(val_psnr):.6f}")
    print(f"val_msssim={float(val_msssim):.8f}")
    print(f"val_dist_1_minus_msssim={float(val_dist):.8f}")
    print(f"val_loss={float(val_loss):.6f}")
    print(f"val_cbr={float(val_cbr):.6f}")
    print(f"csv_saved={csv_path}")


if __name__ == "__main__":
    main()
