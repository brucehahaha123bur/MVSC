'''
python train_mvsc.py \
  --train-root /root/autodl-tmp/Harmony4D/train/01_hugging,/root/autodl-tmp/Harmony4D/train/03_grappling2 \
  --val-root /root/autodl-tmp/Harmony4D/test/01_hugging \
  --output-dir runs/mvsc_jscc_stable \
  --epochs 50 \
  --batch-size 1 \
  --num-workers 2 \
  --lr 1e-5 \
  --num-views 4 \
  --num-frames 4 \
  --crop-size 256 \
  --latent-dim 128 \
  --channel-type awgn \
  --multiple-snr 15 \
  --phase scratch_full \
  --no-amp
'''

import os
import math
import time
import argparse
from pathlib import Path
from torch.utils.data import ConcatDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ====== 你自己的数据集 ======
# 这里请换成你项目里实际的数据集类
# 要求 __getitem__ 返回:
#   x: [T, V, 3, H, W] 或 [B, T, V, 3, H, W] 在 DataLoader 后变成 [B, T, V, 3, H, W]
#
# 示例：
from data.harmony4d_mvsc import Harmony4DMVSCDataset
#
# 下面先留一个占位，你自己替换
# ------------------------------------------------------------
# from your_dataset_file import YourDataset
# ------------------------------------------------------------

from net.encoder2 import (
    MVSC_Individual_Encoder,
    MVSC_Commonality_Encoder,
    MVSC_JSCC_Encoder,
)

from net.decoder2 import (
    MVSC_JSCC_Decoder,
    MVSC_Commonality_Decoder,
    MVSC_Individual_Decoder,
)


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(torch.clamp(mse, min=1e-10))


def save_checkpoint(state: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)


def load_checkpoint(model, optimizer, scheduler, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_psnr = ckpt.get("best_psnr", -1e9)
    return start_epoch, best_psnr


# ============================================================
# Channel
# ============================================================

class AWGNChannel(nn.Module):
    """
    对连续 latent 加 AWGN。
    默认按每个 sample 的平均功率归一化后加噪声。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, snr_db: float):
        # x: [B, D, L, C]
        if snr_db is None:
            return x

        signal_power = torch.mean(x ** 2, dim=(1, 2, 3), keepdim=True)  # [B,1,1,1]
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = torch.sqrt(torch.clamp(noise_power, min=1e-12))

        noise = torch.randn_like(x) * noise_std
        return x + noise


# ============================================================
# Full MVSC model using encoder2 / decoder2
# ============================================================

class MVSCFullModel(nn.Module):
    """
    x: [B, T, V, 3, H, W]
    """
    def __init__(
        self,
        img_size=256,
        patch_size=4,
        in_chans=3,
        out_chans=3,
        num_views=4,
        latent_dim=256,
        individual_depths=(1, 2, 1),
        common_depths=(2, 1),
        common_heads=(8, 10),
        compressed_num_views=2,
        bypass_channel=False,
    ):
        super().__init__()

        self.bypass_channel = bypass_channel
        self.channel = AWGNChannel()
        self.cbr_bits_per_component = 3

        # encoder
        self.individual_encoder = MVSC_Individual_Encoder(
            img_size=img_size,
            patch_size=2,   # encoder2里当前就是按 patch_size=2 设计
            in_chans=in_chans,
            embed_dim=96,
            depths=individual_depths,
        )

        self.commonality_encoder = MVSC_Commonality_Encoder(
            dim=192,
            input_resolution=(img_size // 8, img_size // 8),  # 256 -> 32
            depths=common_depths,
            num_heads=common_heads,
        )

        self.jscc_encoder = MVSC_JSCC_Encoder(
            dim=320,
            latent_dim=latent_dim,
        )

        # decoder
        self.jscc_decoder = MVSC_JSCC_Decoder(
            latent_dim=latent_dim,
            embed_dim=320,
            compressed_num_views=compressed_num_views,
            temporal_upsample_in_jscc=False,
        )

        self.commonality_decoder = MVSC_Commonality_Decoder(
            dim=320,
            input_resolution=(img_size // 8, img_size // 8),
            num_views=num_views,
            compressed_num_views=compressed_num_views,
            depths=(1, 2),
            num_heads=(10, 8),
        )

        self.individual_decoder = MVSC_Individual_Decoder(
            img_size=img_size,
            patch_size=patch_size,   # 这里必须和最终 token-to-image 逻辑匹配
            out_chans=out_chans,
            embed_dim=192,
            input_resolution=(img_size // 8, img_size // 8),
            depths=individual_depths,
        )

    def _compute_cbr(self, y_lat, x):
        """
        Paper-style CBR:
            transmitted_bits / original_source_volume

        y_lat: [B, D_jscc, L_jscc, C_latent]
        x:     [B, T, V, 3, H, W]
        """
        if y_lat.dim() != 4:
            raise ValueError(
                f"_compute_cbr expects y_lat as [B,D,L,C], got shape={tuple(y_lat.shape)}"
            )
        if x.dim() != 6:
            raise ValueError(
                f"_compute_cbr expects x as [B,T,V,3,H,W], got shape={tuple(x.shape)}"
            )

        _, D_jscc, L_jscc, C_latent = y_lat.shape
        _, T, V, C_rgb, H, W = x.shape

        transmitted_bits = D_jscc * L_jscc * C_latent * self.cbr_bits_per_component
        original_volume = T * V * C_rgb * H * W
        return x.new_tensor(float(transmitted_bits) / float(original_volume))

    def forward(self, x, snr_db=None):
        # [B, T, V, 3, H, W]
        y_ind = self.individual_encoder(x)          # [B, T, V, 1024, 192]
        y_com = self.commonality_encoder(y_ind)     # [B, T', V', 1024, 320]
        y_lat = self.jscc_encoder(y_com)            # [B, D, 64, latent_dim]
        cbr = self._compute_cbr(y_lat, x)

        if not self.bypass_channel:
            y_lat = self.channel(y_lat, snr_db)

        y_jscc = self.jscc_decoder(y_lat)           # [B, T', V', 1024, 320]
        y_dec  = self.commonality_decoder(y_jscc)   # [B, T,  V,  1024, 192]
        x_hat  = self.individual_decoder(y_dec)     # [B, T,  V,  3, H, W]

        return x_hat, {
            "y_ind_shape": tuple(y_ind.shape),
            "y_com_shape": tuple(y_com.shape),
            "y_lat_shape": tuple(y_lat.shape),
            "y_jscc_shape": tuple(y_jscc.shape),
            "y_dec_shape": tuple(y_dec.shape),
            "cbr": float(cbr.item()),
        }


# ============================================================
# Train / Val
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    accum_steps=1,
    snr_db=15.0,
    print_freq=20,
    max_norm=1.0,
):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_cbr = 0.0
    num_steps = 0

    optimizer.zero_grad(set_to_none=True)

    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        x = batch["x"]
        x = x.to(device, non_blocking=True).float()

        x_hat, aux = model(x, snr_db=snr_db)
        loss = F.mse_loss(x_hat, x)
        mse = loss.detach()
        psnr = psnr_from_mse(mse)

        (loss / accum_steps).backward()

        if step % accum_steps == 0:
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        running_psnr += psnr.item()
        running_cbr += aux["cbr"]
        num_steps += 1

        if step % print_freq == 0 or step == 1:
            elapsed = time.time() - start_time
            print(
                f"[Train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.item():.6f} avg_loss={running_loss/num_steps:.6f} "
                f"psnr={psnr.item():.3f} avg_psnr={running_psnr/num_steps:.3f} "
                f"cbr={aux['cbr']:.6f} avg_cbr={running_cbr/num_steps:.6f} "
                f"snr={snr_db} time={elapsed:.1f}s"
            )
            if step == 1:
                print(f"  input        : {tuple(x.shape)}")
                print(f"  ind_enc      : {aux['y_ind_shape']}")
                print(f"  common_enc   : {aux['y_com_shape']}")
                print(f"  jscc_latent  : {aux['y_lat_shape']}")
                print(f"  cbr          : {aux['cbr']:.6f}")
                print(f"  jscc_dec     : {aux['y_jscc_shape']}")
                print(f"  common_dec   : {aux['y_dec_shape']}")
                print(f"  output       : {tuple(x_hat.shape)}")

    return (
        running_loss / max(1, num_steps),
        running_psnr / max(1, num_steps),
        running_cbr / max(1, num_steps),
    )


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
    epoch,
    snr_db=15.0,
    print_freq=20,
):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_cbr = 0.0
    num_steps = 0

    for step, batch in enumerate(loader, start=1):
        x = batch["x"]
        x = x.to(device, non_blocking=True).float()

        x_hat, aux = model(x, snr_db=snr_db)
        loss = F.mse_loss(x_hat, x)
        psnr = psnr_from_mse(loss)

        running_loss += loss.item()
        running_psnr += psnr.item()
        running_cbr += aux["cbr"]
        num_steps += 1

        if step % print_freq == 0 or step == 1:
            print(
                f"[Val] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.item():.6f} avg_loss={running_loss/num_steps:.6f} "
                f"psnr={psnr.item():.3f} avg_psnr={running_psnr/num_steps:.3f} "
                f"cbr={aux['cbr']:.6f} avg_cbr={running_cbr/num_steps:.6f} "
                f"snr={snr_db}"
            )

    return (
        running_loss / max(1, num_steps),
        running_psnr / max(1, num_steps),
        running_cbr / max(1, num_steps),
    )


# ============================================================
# Main
# ============================================================

# Helper for parsing comma-separated roots
def parse_root_list(root_spec: str):
    roots = [p.strip() for p in str(root_spec).split(",") if p.strip()]
    if not roots:
        raise ValueError(f"No valid roots parsed from: {root_spec}")
    return roots

def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data-root", "--train-root", dest="data_root", type=str, default=None)
    parser.add_argument("--val-root", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=4)

    # train
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--save-dir", "--output-dir", dest="save_dir", type=str, default="runs/mvsc_e2e")
    parser.add_argument("--resume", type=str, default="")

    # model
    parser.add_argument("--img-size", "--crop-size", dest="img_size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--compressed-num-views", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=4)

    # channel / compatibility with train_jscc.py style
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--multiple-snr", type=float, default=None)
    parser.add_argument("--channel-type", type=str, default="awgn", choices=["awgn", "none"])
    parser.add_argument("--bypass-channel", action="store_true")
    parser.add_argument("--phase", type=str, default="scratch_full")
    parser.add_argument("--no-amp", action="store_true")

    args = parser.parse_args()

    if args.data_root is None:
        parser.error("the following arguments are required: --data-root/--train-root")

    if args.multiple_snr is not None:
        args.snr_db = args.multiple_snr
    if args.snr_db is None:
        args.snr_db = 15.0

    if args.channel_type == "none":
        args.bypass_channel = True

    return args


def main():
    args = get_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    train_roots = parse_root_list(args.data_root)
    val_root = args.val_root if args.val_root else train_roots[0]

    print(f"[Info] train_mvsc.py compatibility mode")
    print(f"[Info] Train uses {len(train_roots)} root(s):")
    for i, r in enumerate(train_roots, start=1):
        print(f"[Info]   [{i}] root={r}")
    print(f"[Info] Val uses root: {val_root}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 使用 Harmony4DMVSCDataset 替换原有数据集占位
    # ========================================================
    if len(train_roots) == 1:
        train_set = Harmony4DMVSCDataset(
            root=train_roots[0],
            num_views=args.num_views,
            num_frames=args.num_frames,
            crop_size=args.img_size,
            resize_shorter_to=None,
            random_crop=True,
            random_flip=True,
            min_common_frames=8,
            repeat=1000,
        )
    else:
        train_sets = []
        for root in train_roots:
            ds = Harmony4DMVSCDataset(
                root=root,
                num_views=args.num_views,
                num_frames=args.num_frames,
                crop_size=args.img_size,
                resize_shorter_to=None,
                random_crop=True,
                random_flip=True,
                min_common_frames=8,
                repeat=1000,
            )
            train_sets.append(ds)
        train_set = ConcatDataset(train_sets)

    val_set = Harmony4DMVSCDataset(
        root=val_root,
        num_views=args.num_views,
        num_frames=args.num_frames,
        crop_size=args.img_size,
        resize_shorter_to=None,
        random_crop=False,
        random_flip=False,
        min_common_frames=8,
        repeat=50,
    )
    # ========================================================

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = MVSCFullModel(
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_views=args.num_views,
        latent_dim=args.latent_dim,
        compressed_num_views=args.compressed_num_views,
        bypass_channel=args.bypass_channel,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    best_psnr = -1e9

    if args.resume:
        start_epoch, best_psnr = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
        print(f"Resume from {args.resume}, start_epoch={start_epoch}, best_psnr={best_psnr:.4f}")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_psnr, train_cbr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            accum_steps=args.accum_steps,
            snr_db=None if args.bypass_channel else args.snr_db,
            print_freq=args.print_freq,
        )

        val_loss, val_psnr, val_cbr = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            snr_db=None if args.bypass_channel else args.snr_db,
            print_freq=max(1, args.print_freq // 2),
        )

        scheduler.step()

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.6f} train_psnr={train_psnr:.3f} train_cbr={train_cbr:.6f} | "
            f"val_loss={val_loss:.6f} val_psnr={val_psnr:.3f} val_cbr={val_cbr:.6f}"
        )

        latest_path = save_dir / "latest.pt"
        save_checkpoint(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_psnr": best_psnr,
                "args": vars(args),
            },
            str(latest_path),
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_path = save_dir / "best_psnr.pt"
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                    "args": vars(args),
                },
                str(best_path),
            )
            print(f"Saved best checkpoint to: {best_path}")


if __name__ == "__main__":
    main()