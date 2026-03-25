import argparse
import math
import os
import random
import time
import tempfile
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from data.harmony4d_mvsc import Harmony4DMVSCDataset
from net import encoder1 as enc_mod
from net import decoder1 as dec_mod


def parse_root_list(root_spec: str) -> List[str]:
    roots: List[str] = []
    for token in str(root_spec).split(","):
        token = token.strip()
        if token:
            roots.append(os.path.abspath(token))
    if not roots:
        raise ValueError("Dataset root specification is empty.")
    return roots


def split_repeat_budget(total_repeat: int, num_parts: int) -> List[int]:
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


def parse_int_tuple(spec: str, expected_len: int, name: str) -> Tuple[int, ...]:
    values = []
    for token in str(spec).split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    if len(values) != expected_len:
        raise ValueError(f"{name} must contain exactly {expected_len} integers, got: {spec}")
    return tuple(values)



def parse_multiple_snr(multiple_snr: str) -> List[float]:
    values = []
    for token in str(multiple_snr).split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("--multiple-snr must contain at least one numeric value.")
    return values


# Helper to parse fixed camera names
def parse_fixed_cams(spec: str | None) -> List[str] | None:
    if spec is None:
        return None
    cams: List[str] = []
    for token in str(spec).split(","):
        token = token.strip()
        if token:
            cams.append(token)
    return cams if cams else None


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModernMVSCNetSimple(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_views = int(args.num_views)
        self.crop_size = int(args.crop_size)
        self.embed_dim = int(args.embed_dim)
        self.individual_dim = 192
        self.common_dim = 320
        self.latent_dim = int(args.latent_dim)
        self.individual_depths = parse_int_tuple(args.individual_depths, 3, "--individual-depths")
        self.common_depths = parse_int_tuple(args.common_depths, 2, "--common-depths")
        self.common_heads = parse_int_tuple(args.common_heads, 2, "--common-heads")
        self.channel_type = str(args.channel_type)
        self.snr_values = parse_multiple_snr(args.multiple_snr)

        self.individual_encoder = enc_mod.MVSC_Individual_Encoder(
            img_size=self.crop_size,
            patch_size=2,
            in_chans=3,
            embed_dim=self.embed_dim,
            depths=self.individual_depths,
        )

        self.common_encoder = enc_mod.MVSC_Commonality_Encoder(
            dim=self.individual_dim,
            input_resolution=(self.crop_size // 8, self.crop_size // 8),
            depths=self.common_depths,
            num_heads=self.common_heads,
        )

        self.jscc_encoder = enc_mod.MVSC_JSCC_Encoder(
            dim=self.common_dim,
            latent_dim=self.latent_dim,
        )

        self.jscc_decoder = dec_mod.MVSC_JSCC_Decoder(
            latent_dim=self.latent_dim,
            embed_dim=self.common_dim,
            compressed_num_views=self.num_views,
            temporal_upsample_in_jscc=False,
        )

    
        self.common_decoder = dec_mod.MVSC_Commonality_Decoder(
            dim=self.common_dim,
            input_resolution=(self.crop_size // 8, self.crop_size // 8),
            num_views=self.num_views,
            depths=self.common_depths,
            num_heads=(self.common_heads[1], self.common_heads[0]),
            out_dim=self.individual_dim,
        )

        self.individual_decoder = dec_mod.MVSC_Individual_Decoder(
            img_size=self.crop_size,
            patch_size=8,
            out_chans=3,
            embed_dim=self.individual_dim,
            input_resolution=(self.crop_size // 8, self.crop_size // 8),
            num_upsample_stages=0,
            depths=self.individual_depths,
        )

    def _sample_snr(self, given_snr=None) -> float:
        if given_snr is not None:
            return float(given_snr)
        return float(random.choice(self.snr_values))

    def _power_normalize(self, y: torch.Tensor) -> torch.Tensor:
        power = torch.mean(y.pow(2), dim=tuple(range(1, y.ndim)), keepdim=True)
        return y / torch.sqrt(power + 1e-8)

    def _apply_channel(self, y: torch.Tensor, snr_db: float) -> torch.Tensor:
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

    def _compute_cbr(self, y: torch.Tensor, x: torch.Tensor, bits_per_component: float) -> torch.Tensor:
        bits = y.shape[1] * y.shape[2] * y.shape[3] * bits_per_component
        denom = x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4] * x.shape[5]
        value = float(bits) / float(denom)
        return x.new_tensor(value)

    def forward(self, x: torch.Tensor, bits_per_component: float = 3.0, given_snr=None):
        x_ind = self.individual_encoder(x)
        x_com = self.common_encoder(x_ind)
        y = self.jscc_encoder(x_com)

        used_snr = self._sample_snr(given_snr)
        y_hat = self._apply_channel(y, used_snr)

        x_jscc = self.jscc_decoder(y_hat)
        x_com_dec = self.common_decoder(x_jscc)
        x_hat = self.individual_decoder(x_com_dec)

        distortion = torch.mean((x_hat - x) ** 2)
        cbr = self._compute_cbr(y, x, bits_per_component=bits_per_component)
        return x_hat, distortion, cbr, used_snr


def build_dataset(root_spec: str, args, is_train: bool):
    resize_shorter_to = args.resize_shorter_to if args.resize_shorter_to > 0 else None
    split_name = "Train" if is_train else "Val"
    root_list = parse_root_list(root_spec)
    repeat_budget = int(args.train_repeat if is_train else args.val_repeat)

    if args.fixed_cams:
        print(f"[Info] build_dataset fixed_cams={args.fixed_cams}")

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
            fixed_cam_names=parse_fixed_cams(args.fixed_cams),
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
                fixed_cam_names=parse_fixed_cams(args.fixed_cams),
            )
        )
    return ConcatDataset(datasets)


def make_dataloader(dataset, args, is_train: bool):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        drop_last=False,
    )


def psnr_from_mse(mse: float) -> float:
    mse = max(float(mse), 1e-12)
    return -10.0 * math.log10(mse)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_val_loss: float, best_val_psnr: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
        "best_val_psnr": best_val_psnr,
    }

    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
        os.close(tmp_fd)
        tmp_fd = None
        torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def train_one_epoch(model, loader, optimizer, device, epoch: int, args):
    model.train()
    running_loss = 0.0
    running_dist = 0.0
    running_cbr = 0.0
    running_psnr = 0.0
    nonfinite_skip_count = 0
    start_time = time.time()

    accum_steps = max(int(args.grad_accum_steps), 1)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)

        x_hat, distortion, cbr, used_snr = model(
            x,
            bits_per_component=args.cbr_bits_per_component,
        )
        loss = distortion + args.cbr_weight * cbr

        if not (torch.isfinite(loss).all() and torch.isfinite(x_hat).all()):
            nonfinite_skip_count += 1
            print(
                f"[Warn] Non-finite forward detected at epoch={epoch} step={step} "
                f"skip_count={nonfinite_skip_count} snr={used_snr}"
            )
            optimizer.zero_grad(set_to_none=True)
            if nonfinite_skip_count > args.max_nonfinite_batches:
                raise RuntimeError(
                    f"Too many non-finite batches in epoch {epoch}: {nonfinite_skip_count} > {args.max_nonfinite_batches}"
                )
            continue

        # 梯度累积：每步只反传 loss/accum_steps
        (loss / accum_steps).backward()

        do_step = (step % accum_steps == 0) or (step == len(loader))
        if do_step:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        mse_value = float(distortion.detach().item())
        loss_value = float(loss.detach().item())
        cbr_value = float(cbr.detach().item())
        psnr_value = psnr_from_mse(mse_value)

        running_loss += loss_value
        running_dist += mse_value
        running_cbr += cbr_value
        running_psnr += psnr_value

        if step == 1 or step % args.log_interval == 0 or step == len(loader):
            elapsed = time.time() - start_time
            avg_loss = running_loss / step
            avg_dist = running_dist / step
            avg_cbr = running_cbr / step
            avg_psnr = running_psnr / step
            print(
                f"[Train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss_value:.6f} avg_loss={avg_loss:.6f} "
                f"dist={mse_value:.6f} avg_dist={avg_dist:.6f} "
                f"cbr={cbr_value:.6f} avg_cbr={avg_cbr:.6f} "
                f"psnr={psnr_value:.3f} avg_psnr={avg_psnr:.3f} "
                f"snr={used_snr:.1f} accum={accum_steps} time={elapsed:.1f}s"
            )

    num_steps = max(len(loader), 1)
    return (
        running_loss / num_steps,
        running_dist / num_steps,
        running_cbr / num_steps,
        running_psnr / num_steps,
        nonfinite_skip_count,
    )


@torch.no_grad()
def evaluate(model, loader, device, epoch: int, args):
    model.eval()
    running_loss = 0.0
    running_dist = 0.0
    running_cbr = 0.0
    running_psnr = 0.0

    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True)
        x_hat, distortion, cbr, used_snr = model(
            x,
            bits_per_component=args.cbr_bits_per_component,
            given_snr=args.eval_snr,
        )
        loss = distortion + args.cbr_weight * cbr

        mse_value = float(distortion.detach().item())
        loss_value = float(loss.detach().item())
        cbr_value = float(cbr.detach().item())
        psnr_value = psnr_from_mse(mse_value)

        running_loss += loss_value
        running_dist += mse_value
        running_cbr += cbr_value
        running_psnr += psnr_value

        if step == 1 or step % args.log_interval == 0 or step == len(loader):
            avg_loss = running_loss / step
            avg_dist = running_dist / step
            avg_cbr = running_cbr / step
            avg_psnr = running_psnr / step
            print(
                f"[Val] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss_value:.6f} avg_loss={avg_loss:.6f} "
                f"dist={mse_value:.6f} avg_dist={avg_dist:.6f} "
                f"cbr={cbr_value:.6f} avg_cbr={avg_cbr:.6f} "
                f"psnr={psnr_value:.3f} avg_psnr={avg_psnr:.3f} "
                f"snr={used_snr:.1f}"
            )

    num_steps = max(len(loader), 1)
    return (
        running_loss / num_steps,
        running_dist / num_steps,
        running_cbr / num_steps,
        running_psnr / num_steps,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=str, required=True)
    parser.add_argument("--val-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="runs/mvsc_simple")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)

    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--min-common-frames", type=int, default=8)
    parser.add_argument("--resize-shorter-to", type=int, default=0)
    parser.add_argument("--train-repeat", type=int, default=2000)
    parser.add_argument("--val-repeat", type=int, default=100)

    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--individual-depths", type=str, default="1,2,1")
    parser.add_argument("--common-depths", type=str, default="2,1")
    parser.add_argument("--common-heads", type=str, default="8,10")

    parser.add_argument("--channel-type", type=str, default="awgn", choices=["none", "awgn", "rayleigh"])
    parser.add_argument("--multiple-snr", type=str, default="15")
    parser.add_argument("--eval-snr", type=float, default=15.0)

    parser.add_argument("--cbr-weight", type=float, default=0.0)
    parser.add_argument("--cbr-bits-per-component", type=float, default=3.0)
    parser.add_argument("--max-nonfinite-batches", type=int, default=20)
    parser.add_argument(
        "--save-interval-epochs",
        type=int,
        default=5,
        help="Save checkpoint every N epochs in addition to best checkpoints. Set 0 to disable.",
    )
    parser.add_argument(
        "--fixed-cams",
        type=str,
        default=None,
        help="Comma-separated camera names, e.g., cam02,cam03,cam11,cam17. If set, dataset will use fixed cameras.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    fixed_cam_names = parse_fixed_cams(args.fixed_cams)
    if fixed_cam_names is not None:
        if len(fixed_cam_names) != int(args.num_views):
            raise ValueError(
                f"--fixed-cams length must equal --num-views: {len(fixed_cam_names)} vs {args.num_views}"
            )
        print(f"[Info] Using fixed cameras: {fixed_cam_names}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Device: {device}")
    print(
        f"[Info] Model config: embed_dim={args.embed_dim}, individual_dim=192, common_dim=320, latent_dim={args.latent_dim}, "
        f"individual_depths={args.individual_depths}, common_depths={args.common_depths}, common_heads={args.common_heads}"
    )
    print(f"[Info] Optimization: batch_size={args.batch_size}, grad_accum_steps={args.grad_accum_steps}")

    train_dataset = build_dataset(args.train_root, args, is_train=True)
    val_dataset = build_dataset(args.val_root, args, is_train=False)
    train_loader = make_dataloader(train_dataset, args, is_train=True)
    val_loader = make_dataloader(val_dataset, args, is_train=False)

    model = ModernMVSCNetSimple(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss_path = output_dir / "best_loss.pt"
    best_psnr_path = output_dir / "best_psnr.pt"

    best_val_loss = float("inf")
    best_val_psnr = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dist, train_cbr, train_psnr, train_nonfinite_skips = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args
        )
        val_loss, val_dist, val_cbr, val_psnr = evaluate(model, val_loader, device, epoch, args)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        if args.save_interval_epochs > 0 and epoch % args.save_interval_epochs == 0:
            periodic_path = output_dir / f"epoch_{epoch}.pt"
            save_checkpoint(
                periodic_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                best_val_psnr,
            )
            print(f"[Info] Saved periodic checkpoint: epoch_{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_loss_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                best_val_psnr,
            )
            print(f"Saved new best checkpoint (epoch-end) with val_loss={val_loss:.6f}")

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            save_checkpoint(
                best_psnr_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                best_val_psnr,
            )
            print(f"Saved new best-psnr checkpoint (epoch-end) with val_psnr={val_psnr:.3f}")

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.6f} train_dist={train_dist:.6f} train_cbr={train_cbr:.6f} train_psnr={train_psnr:.3f} "
            f"val_loss={val_loss:.6f} val_dist={val_dist:.6f} val_cbr={val_cbr:.6f} val_psnr={val_psnr:.3f} "
            f"lr={current_lr:.6e} accum={args.grad_accum_steps} save_interval={args.save_interval_epochs} skipped_nonfinite_train={train_nonfinite_skips}"
        )

    print("Training finished.")


if __name__ == "__main__":
    main()
