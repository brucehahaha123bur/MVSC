import argparse
import csv
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from net.network import MVSCNet
from train_mvsc import (
    build_dataset,
    build_loader,
    evaluate,
    make_config,
    make_model_args,
    resolve_device,
    resolve_resume_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep fixed SNR values and plot MVSC validation statistics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path, e.g. ./runs/exp/best_psnr.pt")
    parser.add_argument("--train-args", type=str, default=None, help="Path to train_args.json (default: sibling of checkpoint)")
    parser.add_argument("--val-root", type=str, default=None, help="Override validation root")
    parser.add_argument("--output-dir", type=str, default="./runs/mvsc_snr_sweep", help="Directory for CSV and plot")
    parser.add_argument("--snr-list", type=str, default="1,4,7,10,13", help="Comma-separated fixed SNR list")
    parser.add_argument("--val-repeat", type=int, default=None, help="Override val_repeat for this sweep")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for sweep (0 recommended for reproducibility)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Override runtime device")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic per-SNR comparison")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Load checkpoint with strict=False when model keys mismatch",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_snr_list(raw: str):
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("--snr-list must contain at least one numeric value")
    return values


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

    return train_args, train_args_path


def plot_results(rows, save_path, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(
            "[Warn] matplotlib is not installed. Skipping plot. "
            "Install it with: pip install matplotlib"
        )
        print(f"[Warn] reason={exc}")
        return False

    snrs = [r["snr"] for r in rows]
    psnrs = [r["val_psnr"] for r in rows]
    losses = [r["val_loss"] for r in rows]
    dists = [r["val_dist"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=140)
    ax1.plot(snrs, psnrs, marker="o", linewidth=2.0, color="#1f77b4", label="PSNR")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("PSNR (dB)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(snrs, losses, marker="s", linestyle="--", linewidth=1.8, color="#d62728", label="Val Loss")
    ax2.plot(snrs, dists, marker="^", linestyle="-.", linewidth=1.6, color="#ff7f0e", label="Val Dist")
    ax2.set_ylabel("Loss / Distortion", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def main():
    args = parse_args()

    set_seed(args.seed)

    checkpoint_path = resolve_resume_path(args.checkpoint, output_dir=os.path.join(ROOT_DIR, "runs"))
    train_args, train_args_path = load_train_args(args, checkpoint_path)

    if args.device is not None:
        train_args.device = args.device
    if args.val_root is not None:
        train_args.val_root = args.val_root
    if args.val_repeat is not None:
        train_args.val_repeat = int(args.val_repeat)

    train_args.num_workers = int(args.num_workers)
    train_args.log_interval = int(1e9)

    snr_values = parse_snr_list(args.snr_list)
    val_root = train_args.val_root if train_args.val_root is not None else train_args.train_root

    device = resolve_device(train_args.device)
    if train_args.channel_type in {"awgn", "rayleigh"} and device.type != "cuda":
        raise ValueError("AWGN/Rayleigh sweep requires CUDA. Use --device cuda or switch channel_type to none.")

    model = MVSCNet(make_model_args(train_args), make_config(train_args, device)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    incompatible = model.load_state_dict(state_dict, strict=not args.allow_mismatch)
    if args.allow_mismatch:
        if incompatible.missing_keys:
            print(f"[Warn] Missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"[Warn] Unexpected keys: {len(incompatible.unexpected_keys)}")

    model.eval()

    val_dataset = build_dataset(val_root, train_args, is_train=False)
    val_loader = build_loader(val_dataset, train_args, is_train=False, device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "snr_sweep.csv")
    png_path = os.path.join(args.output_dir, "snr_sweep.png")

    print(f"checkpoint={checkpoint_path}")
    print(f"train_args={train_args_path}")
    print(f"val_root={val_root}")
    print(f"device={device}")
    print(f"val_repeat={train_args.val_repeat}")
    print(f"snr_list={snr_values}")

    rows = []
    for snr in snr_values:
        # Reset RNG so all SNRs are evaluated on the same sampled validation sequence.
        set_seed(args.seed)
        val_loss, val_psnr, val_dist, val_cbr = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            args=train_args,
            epoch=0,
            given_snr_override=float(snr),
        )
        row = {
            "snr": float(snr),
            "val_psnr": float(val_psnr),
            "val_loss": float(val_loss),
            "val_dist": float(val_dist),
            "val_cbr": float(val_cbr),
        }
        rows.append(row)
        print(
            f"SNR={row['snr']:.2f} "
            f"val_psnr={row['val_psnr']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_dist={row['val_dist']:.4f} "
            f"val_cbr={row['val_cbr']:.6f}"
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["snr", "val_psnr", "val_loss", "val_dist", "val_cbr"])
        writer.writeheader()
        writer.writerows(rows)

    title = f"MVSC SNR Sweep ({os.path.basename(checkpoint_path)})"
    plotted = plot_results(rows, png_path, title=title)

    print(f"csv_saved={csv_path}")
    if plotted:
        print(f"plot_saved={png_path}")


if __name__ == "__main__":
    main()
