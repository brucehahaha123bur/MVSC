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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from net.network import MVSCNet
from train_mvsc import build_dataset, build_loader, make_config, make_model_args, resolve_device, resolve_resume_path


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


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep effective CBR under fixed SNR and plot MVSC validation statistics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path, e.g. ./runs/exp/best_psnr.pt")
    parser.add_argument("--train-args", type=str, default=None, help="Path to train_args.json (default: sibling of checkpoint)")
    parser.add_argument("--val-root", type=str, default=None, help="Override validation root")
    parser.add_argument("--output-dir", type=str, default="./runs/mvsc_cbr_sweep", help="Directory for CSV and plot")
    parser.add_argument("--snr", type=float, default=10.0, help="Fixed SNR used for all CBR sweep points")
    parser.add_argument(
        "--rate-ratios",
        type=str,
        default="0.25,0.5,0.75,1.0",
        help="Comma-separated keep ratios for latent channels; each ratio in (0,1]",
    )
    parser.add_argument("--val-repeat", type=int, default=None, help="Override val_repeat for this sweep")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for sweep (0 recommended for reproducibility)")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Override runtime device")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic per-ratio comparison")
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


def parse_ratio_list(raw: str):
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            value = float(token)
            if value <= 0 or value > 1:
                raise ValueError("Each keep ratio must satisfy 0 < ratio <= 1")
            values.append(value)
    if not values:
        raise ValueError("--rate-ratios must contain at least one ratio")

    values = sorted(set(values))
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
    if not hasattr(train_args, "cbr_weight"):
        train_args.cbr_weight = 8192.0
    if not hasattr(train_args, "cbr_bits_per_component"):
        train_args.cbr_bits_per_component = 3.0

    return train_args, train_args_path


def compute_psnr(x_hat, x):
    mse = torch.mean((x_hat - x) ** 2).item()
    mse = max(mse, 1e-12)
    return -10.0 * math.log10(mse)


def forward_with_rate_ratio(model, x_g, snr_db: float, keep_ratio: float):
    # Encode to latent tokens.
    _, _, y = model.encode(x_g)

    latent_dim = int(y.shape[-1])
    keep_dim = max(1, min(latent_dim, int(round(latent_dim * keep_ratio))))

    # Transmit only the first keep_dim channels and zero-pad back for decoder.
    y_tx = y[..., :keep_dim]
    y_hat_tx = model.channel.forward(y_tx, float(snr_db))

    if keep_dim < latent_dim:
        pad_shape = list(y_hat_tx.shape)
        pad_shape[-1] = latent_dim - keep_dim
        y_pad = torch.zeros(pad_shape, dtype=y_hat_tx.dtype, device=y_hat_tx.device)
        y_hat = torch.cat([y_hat_tx, y_pad], dim=-1)
    else:
        y_hat = y_hat_tx

    # Decode and compute losses/metrics.
    _, _, x_hat = model.decode(y_hat)
    x_hat = x_hat.clamp(0.0, 1.0)

    if x_g.dim() == 6:
        bsz, t, v, c, h, w = x_g.shape
        x_g_ = x_g.reshape(bsz * t * v, c, h, w)
        x_hat_ = x_hat.reshape(bsz * t * v, c, h, w)
    else:
        x_g_ = x_g
        x_hat_ = x_hat

    distortion = model.distortion_loss.forward(x_g_, x_hat_, normalization=model.config.norm)
    source_values = float(x_g.numel())
    bits_per_component = float(getattr(model.args, "cbr_bits_per_component", 3.0))
    transmitted_bits = float(y_tx.numel()) * bits_per_component
    cbr = x_hat.new_tensor(transmitted_bits / max(source_values, 1.0))
    loss = distortion + model.cbr_weight * cbr

    aux = {
        "distortion": distortion.detach(),
        "cbr": cbr.detach(),
        "keep_dim": keep_dim,
        "latent_dim": latent_dim,
    }
    return x_hat, loss, aux


def evaluate_with_rate_ratio(model, loader, device, snr_db, keep_ratio):
    model.eval()
    loss_meter = AverageMeter()
    distortion_meter = AverageMeter()
    cbr_meter = AverageMeter()
    psnr_meter = AverageMeter()

    keep_dim_any = None
    latent_dim_any = None

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            x_hat, loss, aux = forward_with_rate_ratio(model, x, snr_db=snr_db, keep_ratio=keep_ratio)

            distortion_value = float(aux["distortion"].item())
            cbr_value = float(aux["cbr"].item())
            keep_dim_any = int(aux["keep_dim"])
            latent_dim_any = int(aux["latent_dim"])

            bsz = x.shape[0]
            loss_meter.update(loss.item(), bsz)
            distortion_meter.update(distortion_value, bsz)
            cbr_meter.update(cbr_value, bsz)
            psnr_meter.update(compute_psnr(x_hat, x), bsz)

    return (
        loss_meter.avg,
        psnr_meter.avg,
        distortion_meter.avg,
        cbr_meter.avg,
        keep_dim_any,
        latent_dim_any,
    )


def plot_results(rows, save_path, title):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("[Warn] matplotlib is not installed. Skipping plot. Install with: pip install matplotlib")
        print(f"[Warn] reason={exc}")
        return False

    rows = sorted(rows, key=lambda r: r["val_cbr"])
    cbrs = [r["val_cbr"] for r in rows]
    psnrs = [r["val_psnr"] for r in rows]
    losses = [r["val_loss"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=140)
    ax1.plot(cbrs, psnrs, marker="o", linewidth=2.0, color="#1f77b4", label="PSNR")
    ax1.set_xlabel("CBR")
    ax1.set_ylabel("PSNR (dB)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(cbrs, losses, marker="s", linestyle="--", linewidth=1.8, color="#d62728", label="Val Loss")
    ax2.set_ylabel("Loss", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    for row in rows:
        ax1.annotate(
            f"r={row['keep_ratio']:.2f}",
            (row["val_cbr"], row["val_psnr"]),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=8,
        )

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

    keep_ratios = parse_ratio_list(args.rate_ratios)
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
    csv_path = os.path.join(args.output_dir, "cbr_sweep.csv")
    png_path = os.path.join(args.output_dir, "cbr_sweep.png")

    print(f"checkpoint={checkpoint_path}")
    print(f"train_args={train_args_path}")
    print(f"val_root={val_root}")
    print(f"device={device}")
    print(f"val_repeat={train_args.val_repeat}")
    print(f"snr={float(args.snr):.2f}")
    print(f"cbr_bits_per_component={float(getattr(train_args, 'cbr_bits_per_component', 3.0)):.3f}")
    print(f"keep_ratios={keep_ratios}")

    rows = []
    for keep_ratio in keep_ratios:
        # Reset RNG so all ratio points see the same sampled validation sequence.
        set_seed(args.seed)
        val_loss, val_psnr, val_dist, val_cbr, keep_dim, latent_dim = evaluate_with_rate_ratio(
            model=model,
            loader=val_loader,
            device=device,
            snr_db=float(args.snr),
            keep_ratio=float(keep_ratio),
        )

        row = {
            "keep_ratio": float(keep_ratio),
            "keep_dim": int(keep_dim),
            "latent_dim": int(latent_dim),
            "snr": float(args.snr),
            "val_cbr": float(val_cbr),
            "val_psnr": float(val_psnr),
            "val_loss": float(val_loss),
            "val_dist": float(val_dist),
        }
        rows.append(row)

        print(
            f"ratio={row['keep_ratio']:.2f} "
            f"keep={row['keep_dim']}/{row['latent_dim']} "
            f"cbr={row['val_cbr']:.6f} "
            f"psnr={row['val_psnr']:.4f} "
            f"loss={row['val_loss']:.4f} "
            f"dist={row['val_dist']:.4f}"
        )

    rows = sorted(rows, key=lambda r: r["val_cbr"])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["keep_ratio", "keep_dim", "latent_dim", "snr", "val_cbr", "val_psnr", "val_loss", "val_dist"],
        )
        writer.writeheader()
        writer.writerows(rows)

    title = f"MVSC CBR Sweep @ SNR={float(args.snr):.1f} ({os.path.basename(checkpoint_path)})"
    plotted = plot_results(rows, png_path, title=title)

    print(f"csv_saved={csv_path}")
    if plotted:
        print(f"plot_saved={png_path}")


if __name__ == "__main__":
    main()
