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
    parser = argparse.ArgumentParser(
        description="Strict CBR sweep across multiple trained checkpoints (no runtime latent truncation)"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        required=True,
        help=(
            "Comma-separated checkpoint paths. "
            "Each checkpoint is evaluated with its own train_args-derived architecture."
        ),
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional comma-separated labels for checkpoints (same length as --checkpoints)",
    )
    parser.add_argument("--val-root", type=str, default=None, help="Override validation root for all checkpoints")
    parser.add_argument("--output-dir", type=str, default="./runs/mvsc_cbr_sweep_strict", help="Directory for CSV and plot")
    parser.add_argument("--snr", type=float, default=10.0, help="Fixed SNR for all checkpoints")
    parser.add_argument("--val-repeat", type=int, default=80, help="Validation repeat for all checkpoints")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 recommended)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Runtime device")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic comparison")
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip incompatible/broken checkpoints instead of aborting the entire sweep",
    )
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


def parse_csv_list(raw: str):
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            values.append(token)
    return values


def load_train_args_for_checkpoint(ckpt_path: str):
    train_args_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "train_args.json")
    if not os.path.isfile(train_args_path):
        raise FileNotFoundError(
            "train_args.json not found beside checkpoint. "
            f"checkpoint={ckpt_path}, expected={train_args_path}"
        )

    with open(train_args_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    args = SimpleNamespace(**raw)

    # Backward compatibility for old runs.
    if not hasattr(args, "repeat_per_exo"):
        args.repeat_per_exo = False
    if not hasattr(args, "max_nonfinite_batches"):
        args.max_nonfinite_batches = 20
    if not hasattr(args, "cbr_weight"):
        args.cbr_weight = 8192.0

    return args, train_args_path


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

    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    ax.plot(cbrs, psnrs, marker="o", linewidth=2.0, color="#1f77b4")
    ax.set_xlabel("CBR")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    for row in rows:
        ax.annotate(
            row["label"],
            (row["val_cbr"], row["val_psnr"]),
            textcoords="offset points",
            xytext=(5, 6),
            fontsize=8,
        )

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def evaluate_one_checkpoint(ckpt_path, label, args):
    train_args, train_args_path = load_train_args_for_checkpoint(ckpt_path)

    # Unified evaluation controls for fairness.
    train_args.device = args.device
    train_args.val_repeat = int(args.val_repeat)
    train_args.num_workers = int(args.num_workers)
    train_args.log_interval = int(1e9)
    if args.val_root is not None:
        train_args.val_root = args.val_root

    val_root = train_args.val_root if train_args.val_root is not None else train_args.train_root

    device = resolve_device(train_args.device)
    if train_args.channel_type in {"awgn", "rayleigh"} and device.type != "cuda":
        raise ValueError("AWGN/Rayleigh evaluation requires CUDA. Use --device cuda or channel_type none.")

    model = MVSCNet(make_model_args(train_args), make_config(train_args, device)).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    try:
        incompatible = model.load_state_dict(state_dict, strict=not args.allow_mismatch)
    except RuntimeError as exc:
        raise RuntimeError(
            "state_dict mismatch: checkpoint architecture is incompatible with current MVSCNet definition. "
            "Use a checkpoint from the same architecture family, or pass --allow-mismatch for exploratory runs."
        ) from exc
    if args.allow_mismatch:
        if incompatible.missing_keys:
            print(f"[Warn] {label}: missing keys={len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"[Warn] {label}: unexpected keys={len(incompatible.unexpected_keys)}")

    model.eval()

    val_dataset = build_dataset(val_root, train_args, is_train=False)
    val_loader = build_loader(val_dataset, train_args, is_train=False, device=device)

    # Reset RNG before each checkpoint so sampled validation sequence is aligned.
    set_seed(args.seed)
    val_loss, val_psnr, val_dist, val_cbr = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        args=train_args,
        epoch=0,
        given_snr_override=float(args.snr),
    )

    latent_dim = int(getattr(train_args, "latent_dim", -1))

    row = {
        "label": label,
        "checkpoint": ckpt_path,
        "train_args": train_args_path,
        "latent_dim": latent_dim,
        "snr": float(args.snr),
        "val_cbr": float(val_cbr),
        "val_psnr": float(val_psnr),
        "val_loss": float(val_loss),
        "val_dist": float(val_dist),
        "val_root": val_root,
    }
    return row


def main():
    args = parse_args()
    set_seed(args.seed)

    checkpoint_inputs = parse_csv_list(args.checkpoints)
    if len(checkpoint_inputs) == 0:
        raise ValueError("--checkpoints must contain at least one path")

    labels = parse_csv_list(args.labels) if args.labels else []
    if labels and len(labels) != len(checkpoint_inputs):
        raise ValueError("--labels length must equal --checkpoints length")

    resolved = []
    for raw_ckpt in checkpoint_inputs:
        ckpt_path = resolve_resume_path(raw_ckpt, output_dir=os.path.join(ROOT_DIR, "runs"))
        resolved.append(ckpt_path)

    if not labels:
        labels = [
            os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
            for path in resolved
        ]

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "cbr_sweep_strict.csv")
    png_path = os.path.join(args.output_dir, "cbr_sweep_strict.png")

    print(f"snr={float(args.snr):.2f}")
    print(f"val_repeat={int(args.val_repeat)}")
    print(f"device={args.device}")
    if args.val_root is not None:
        print(f"val_root_override={args.val_root}")

    rows = []
    for idx, (ckpt_path, label) in enumerate(zip(resolved, labels), start=1):
        print(f"[Info] ({idx}/{len(resolved)}) evaluating {label}")
        try:
            row = evaluate_one_checkpoint(ckpt_path=ckpt_path, label=label, args=args)
            rows.append(row)
            print(
                f"[Result] label={row['label']} latent_dim={row['latent_dim']} "
                f"cbr={row['val_cbr']:.6f} psnr={row['val_psnr']:.4f} "
                f"loss={row['val_loss']:.4f} dist={row['val_dist']:.4f}"
            )
        except Exception as exc:
            if args.skip_failed:
                print(f"[Warn] skip failed checkpoint: {label}. reason={exc}")
                continue
            raise

    if len(rows) == 0:
        raise RuntimeError("No valid checkpoints were evaluated. Use --skip-failed and provide compatible checkpoints.")

    rows = sorted(rows, key=lambda r: r["val_cbr"])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "checkpoint",
                "train_args",
                "latent_dim",
                "snr",
                "val_cbr",
                "val_psnr",
                "val_loss",
                "val_dist",
                "val_root",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    title = f"MVSC Strict CBR Sweep @ SNR={float(args.snr):.1f}"
    plotted = plot_results(rows, png_path, title)

    print(f"csv_saved={csv_path}")
    if plotted:
        print(f"plot_saved={png_path}")


if __name__ == "__main__":
    main()
