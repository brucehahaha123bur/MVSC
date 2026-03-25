import os
import csv
import argparse
import torch

from train_jscc import (
    set_seed,
    resolve_device,
    build_dataset,
    build_loader,
    make_model_args,
    ModernMVSCNet,
    evaluate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="runs/eval_snr_sweep.csv")
    parser.add_argument("--snr-list", type=str, default="0,5,10,15,20")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--min-common-frames", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--resize-shorter-to", type=int, default=0)
    parser.add_argument("--center-crop-size", type=int, default=0)
    parser.add_argument("--train-repeat", type=int, default=2000)
    parser.add_argument("--val-repeat", type=int, default=100)

    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--individual-depths", type=str, default="1,2,1")
    parser.add_argument("--common-depths", type=str, default="2,1")
    parser.add_argument("--common-heads", type=str, default="8,10")

    parser.add_argument("--channel-type", type=str, default="awgn")
    parser.add_argument("--distortion-metric", type=str, default="MSE")
    parser.add_argument("--cbr-weight", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true", default=False)

    # 下面这些字段只是为了和 train_jscc 的 args 兼容
    parser.add_argument("--phase", type=str, default="full_finetune")
    parser.add_argument("--freeze-individual", action="store_true", default=False)
    parser.add_argument("--freeze-commonality", action="store_true", default=False)
    parser.add_argument("--train-root", type=str, default="")
    parser.add_argument("--val-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/eval_tmp")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--multiple-snr", type=str, default="15")
    parser.add_argument("--amp-dtype", type=str, default="float16")
    parser.add_argument("--cbr-bits-per-component", type=float, default=3.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--persistent-workers", action="store_true", default=False)

    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # 只建 val dataset
    val_dataset = build_dataset(args.data_root, args, is_train=False)
    val_loader = build_loader(val_dataset, args, is_train=False, device=device)

    model_args = make_model_args(args)
    model = ModernMVSCNet(model_args).to(device)
    model.channel_type = args.channel_type

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    snr_list = [float(x.strip()) for x in args.snr_list.split(",") if x.strip() != ""]

    rows = []
    for snr in snr_list:
        val_loss, val_psnr, val_dist, val_cbr = evaluate(
            model,
            val_loader,
            device,
            args,
            epoch=0,
            given_snr_override=snr,
        )
        print(
            f"[SNR Sweep] snr={snr:.1f} "
            f"val_loss={val_loss:.6f} val_dist={val_dist:.6f} "
            f"val_cbr={val_cbr:.6f} val_psnr={val_psnr:.3f}"
        )
        rows.append({
            "snr": snr,
            "val_loss": val_loss,
            "val_dist": val_dist,
            "val_cbr": val_cbr,
            "val_psnr": val_psnr,
        })

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["snr", "val_loss", "val_dist", "val_cbr", "val_psnr"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Done] Saved to {args.output_csv}")


if __name__ == "__main__":
    main()