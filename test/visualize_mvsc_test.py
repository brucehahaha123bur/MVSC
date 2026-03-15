import argparse
import csv
import json
import math
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from net.network import MVSCNet
from train_mvsc import (
    build_dataset,
    make_config,
    make_model_args,
    parse_multiple_snr,
    resolve_device,
    resolve_resume_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visual test for MVSC reconstruction quality")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path, e.g. ./runs/exp/best.pt")
    parser.add_argument("--train-args", type=str, default=None, help="Path to train_args.json")
    parser.add_argument("--train-root", type=str, default=None, help="Override train root")
    parser.add_argument("--val-root", type=str, default=None, help="Override val root")
    parser.add_argument("--output-dir", type=str, default="./runs/mvsc_visual_test", help="Output directory")
    parser.add_argument("--split", type=str, default="both", choices=["train", "val", "both"])
    parser.add_argument("--num-samples", type=int, default=4, help="Samples to visualize per split")
    parser.add_argument("--show-frames", type=int, default=4, help="Number of frames shown per sample")
    parser.add_argument("--show-views", type=int, default=4, help="Number of views shown per sample")
    parser.add_argument("--tile-size", type=int, default=128, help="Tile size for visualization")
    parser.add_argument("--given-snr", type=float, default=None, help="Override SNR for visual test")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_train_args(args):
    if args.train_args is not None:
        train_args_path = args.train_args
    else:
        ckpt_abs = os.path.abspath(args.checkpoint)
        run_dir = os.path.dirname(ckpt_abs)
        train_args_path = os.path.join(run_dir, "train_args.json")

    if not os.path.isfile(train_args_path):
        raise FileNotFoundError(
            "train_args.json not found. Please pass --train-args explicitly. "
            f"Tried: {train_args_path}"
        )

    with open(train_args_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return SimpleNamespace(**raw), train_args_path


def resolve_eval_snr(args, train_args):
    if args.given_snr is not None:
        return float(args.given_snr)
    if getattr(train_args, "val_given_snr", None) is not None:
        return float(train_args.val_given_snr)
    if getattr(train_args, "given_snr", None) is not None:
        return float(train_args.given_snr)
    snr_values = parse_multiple_snr(getattr(train_args, "multiple_snr", "10"))
    return float(snr_values[0])


def uint8_from_tensor(chw):
    arr = chw.detach().cpu().clamp(0.0, 1.0).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return arr


def error_map_uint8(chw_in, chw_out, scale=5.0):
    x = chw_in.detach().cpu().clamp(0.0, 1.0).numpy()
    y = chw_out.detach().cpu().clamp(0.0, 1.0).numpy()
    err = np.mean(np.abs(x - y), axis=0)
    err = np.clip(err * scale, 0.0, 1.0)

    r = (err * 255.0 + 0.5).astype(np.uint8)
    g = (np.clip(1.0 - err, 0.0, 1.0) * 64.0 + 0.5).astype(np.uint8)
    b = np.zeros_like(r, dtype=np.uint8)
    return np.stack([r, g, b], axis=-1)


def make_grid(tile_rows, tile_size=128, gap=6, bg=(24, 24, 24)):
    rows = len(tile_rows)
    cols = len(tile_rows[0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        raise ValueError("Empty tile grid.")

    canvas_w = cols * tile_size + (cols - 1) * gap
    canvas_h = rows * tile_size + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg)

    for r in range(rows):
        for c in range(cols):
            tile = tile_rows[r][c]
            tile_img = Image.fromarray(tile).resize((tile_size, tile_size), resample=Image.BICUBIC)
            x0 = c * (tile_size + gap)
            y0 = r * (tile_size + gap)
            canvas.paste(tile_img, (x0, y0))

    return canvas


def compose_panel(input_grid, recon_grid, error_grid, info_lines):
    margin = 16
    col_gap = 16
    title_h = 28
    info_h = 18 * len(info_lines) + 16

    col_w = max(input_grid.width, recon_grid.width, error_grid.width)
    col_h = max(input_grid.height, recon_grid.height, error_grid.height)

    total_w = margin * 2 + col_w * 3 + col_gap * 2
    total_h = margin * 2 + info_h + title_h + col_h
    canvas = Image.new("RGB", (total_w, total_h), color=(14, 16, 20))
    draw = ImageDraw.Draw(canvas)

    y_text = margin
    for line in info_lines:
        draw.text((margin, y_text), line, fill=(230, 230, 230))
        y_text += 18

    title_y = margin + info_h
    x_in = margin
    x_re = margin + col_w + col_gap
    x_er = margin + (col_w + col_gap) * 2

    draw.text((x_in, title_y), "Input", fill=(255, 255, 255))
    draw.text((x_re, title_y), "Reconstruction", fill=(255, 255, 255))
    draw.text((x_er, title_y), "Abs Error (heat)", fill=(255, 255, 255))

    y_img = title_y + title_h
    canvas.paste(input_grid, (x_in, y_img))
    canvas.paste(recon_grid, (x_re, y_img))
    canvas.paste(error_grid, (x_er, y_img))

    return canvas


def build_tiles(x, x_hat, show_frames, show_views):
    # x/x_hat: [1, T, V, 3, H, W]
    xt = x[0]
    yt = x_hat[0]
    T = min(show_frames, xt.shape[0])
    V = min(show_views, xt.shape[1])

    input_rows = []
    recon_rows = []
    error_rows = []

    for t in range(T):
        in_row = []
        re_row = []
        er_row = []
        for v in range(V):
            in_tile = uint8_from_tensor(xt[t, v])
            re_tile = uint8_from_tensor(yt[t, v])
            er_tile = error_map_uint8(xt[t, v], yt[t, v])
            in_row.append(in_tile)
            re_row.append(re_tile)
            er_row.append(er_tile)
        input_rows.append(in_row)
        recon_rows.append(re_row)
        error_rows.append(er_row)

    return input_rows, recon_rows, error_rows


def forward_once(model, sample, device, eval_snr):
    x = sample["x"].unsqueeze(0).to(device)
    with torch.no_grad():
        model_out = model(x, given_SNR=eval_snr)

    if isinstance(model_out, tuple) and len(model_out) == 4:
        x_hat, used_snr, loss, aux = model_out
        distortion_value = float(aux["distortion"].item()) if "distortion" in aux else float(loss.item())
        cbr_value = float(aux["cbr"].item()) if "cbr" in aux else 0.0
    else:
        x_hat, used_snr, loss = model_out
        distortion_value = float(loss.item())
        cbr_value = 0.0

    mse = torch.mean((x_hat - x) ** 2).item()
    mse = max(mse, 1e-12)
    psnr = -10.0 * math.log10(mse)

    metrics = {
        "loss": float(loss.item()),
        "distortion": distortion_value,
        "cbr": cbr_value,
        "psnr": float(psnr),
        "used_snr": float(used_snr),
    }
    return x, x_hat, metrics


def visualize_split(model, ds, split_name, out_dir, args, eval_snr):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"summary_{split_name}.csv")

    rows = []
    for i in range(args.num_samples):
        sample = ds[i]
        x, x_hat, metrics = forward_once(model, sample, args.runtime_device, eval_snr)
        in_rows, re_rows, er_rows = build_tiles(x, x_hat, args.show_frames, args.show_views)

        input_grid = make_grid(in_rows, tile_size=args.tile_size)
        recon_grid = make_grid(re_rows, tile_size=args.tile_size)
        error_grid = make_grid(er_rows, tile_size=args.tile_size)

        cam_names = sample.get("cam_names", [])
        frame_names = sample.get("frame_names", [])
        info_lines = [
            f"split={split_name} sample={i} checkpoint={os.path.basename(args.checkpoint)}",
            "cams=" + ",".join(cam_names),
            "frames=" + ",".join(frame_names),
            (
                f"loss={metrics['loss']:.6f} dist={metrics['distortion']:.6f} "
                f"cbr={metrics['cbr']:.6f} psnr={metrics['psnr']:.3f} snr={metrics['used_snr']:.2f}"
            ),
        ]

        panel = compose_panel(input_grid, recon_grid, error_grid, info_lines)
        save_path = os.path.join(out_dir, f"{split_name}_sample_{i:03d}.png")
        panel.save(save_path)

        row = {
            "split": split_name,
            "sample": i,
            "loss": metrics["loss"],
            "distortion": metrics["distortion"],
            "cbr": metrics["cbr"],
            "psnr": metrics["psnr"],
            "snr": metrics["used_snr"],
            "cams": ",".join(cam_names),
            "frames": ",".join(frame_names),
            "image": os.path.basename(save_path),
        }
        rows.append(row)
        print(
            f"[{split_name}] sample={i} loss={metrics['loss']:.6f} "
            f"dist={metrics['distortion']:.6f} cbr={metrics['cbr']:.6f} psnr={metrics['psnr']:.3f}"
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "sample",
                "loss",
                "distortion",
                "cbr",
                "psnr",
                "snr",
                "cams",
                "frames",
                "image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        avg_loss = sum(r["loss"] for r in rows) / len(rows)
        avg_psnr = sum(r["psnr"] for r in rows) / len(rows)
        print(f"[{split_name}] avg_loss={avg_loss:.6f}, avg_psnr={avg_psnr:.3f}, csv={csv_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    train_args, train_args_path = load_train_args(args)
    if args.train_root is not None:
        train_args.train_root = args.train_root
    if args.val_root is not None:
        train_args.val_root = args.val_root

    run_dir = os.path.dirname(os.path.abspath(train_args_path))
    args.checkpoint = resolve_resume_path(args.checkpoint, output_dir=run_dir)

    runtime_device = resolve_device(args.device)
    args.runtime_device = runtime_device

    eval_snr = resolve_eval_snr(args, train_args)

    print(f"train_args={train_args_path}")
    print(f"checkpoint={args.checkpoint}")
    print(f"device={runtime_device}")
    print(f"eval_snr={eval_snr}")

    if getattr(train_args, "channel_type", "none") in {"awgn", "rayleigh"} and runtime_device.type != "cuda":
        raise ValueError("AWGN/Rayleigh visual test requires CUDA. Use --device cuda or channel-type none.")

    model = MVSCNet(make_model_args(train_args), make_config(train_args, runtime_device)).to(runtime_device)
    checkpoint = torch.load(args.checkpoint, map_location=runtime_device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.split in {"train", "both"}:
        train_ds = build_dataset(train_args.train_root, train_args, is_train=False)
        out_dir = os.path.join(args.output_dir, "train")
        visualize_split(model, train_ds, "train", out_dir, args, eval_snr)

    if args.split in {"val", "both"}:
        val_root = train_args.val_root if train_args.val_root is not None else train_args.train_root
        if train_args.val_root is None:
            print("[Warn] val_root is None. Validation visual test uses train_root.")
        val_ds = build_dataset(val_root, train_args, is_train=False)
        out_dir = os.path.join(args.output_dir, "val")
        visualize_split(model, val_ds, "val", out_dir, args, eval_snr)

    print("Visualization test finished.")


if __name__ == "__main__":
    main()
