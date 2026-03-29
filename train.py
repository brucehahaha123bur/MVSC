import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
import argparse
import json
import math
import os
import random
import time
import glob
import sys
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import ConcatDataset, DataLoader

# ====== 全局collate_fn，跳过None样本 ======
def skip_none_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

from data.harmony4d_mvsc import Harmony4DMVSCDataset


# =========================================================
# 0. Defaults
# =========================================================
DEFAULT_HARMONY_TRAIN_ROOT = "E:/Harmony4D/train"
DEFAULT_HARMONY_VAL_ROOT = "E:/Harmony4D/test"


# =========================================================
# 1. AMP compatibility
# =========================================================
try:
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast as _autocast
    _HAS_NEW_AMP = True
except ImportError:
    from torch.cuda.amp import GradScaler as _GradScaler
    from torch.cuda.amp import autocast as _autocast
    _HAS_NEW_AMP = False


# =========================================================
# 2. Dynamic import for latest encoder/decoder copy
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
ENCODER_COPY_PATH = PROJECT_ROOT.parent / "net" / "encoder copy.py"
DECODER_COPY_PATH = PROJECT_ROOT.parent / "net" / "decoder copy.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_module_as(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


enc_mod = load_module_as("net.encoder", ENCODER_COPY_PATH)
dec_mod = load_module_as("net.decoder", DECODER_COPY_PATH)


# =========================================================
# 3. Helpers
# =========================================================
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


def resolve_device(device_name: str):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def compute_psnr(x_hat, x):
    mse = torch.mean((x_hat - x) ** 2).item()
    mse = max(mse, 1e-12)
    return -10.0 * math.log10(mse)


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

    basename_only = resume_path and os.path.basename(resume_path) == resume_path
    if basename_only:
        runs_root = os.path.join(os.getcwd(), "runs")
        pattern = os.path.join(runs_root, "**", resume_path)
        matches = [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]
        if matches:
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

    tried = "\n".join(f"  - {p}" for p in unique_candidates)
    raise FileNotFoundError(f"Resume checkpoint not found. Tried:\n{tried}")


# =========================================================
# 4. Dataset discovery
# =========================================================
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
        collate_fn=skip_none_collate,
    )


# =========================================================
# 5. Model wrapper using encoder/decoder copy
# =========================================================
class SimpleChannel(nn.Module):
    def __init__(self, channel_type="awgn"):
        super().__init__()
        self.channel_type = channel_type

    @staticmethod
    def power_normalize(x, eps=1e-8):
        dims = tuple(range(1, x.dim()))
        power = torch.mean(x ** 2, dim=dims, keepdim=True)
        return x / torch.sqrt(power + eps)

    def awgn(self, x, snr_db):
        snr_linear = 10 ** (float(snr_db) / 10.0)
        noise_var = 1.0 / snr_linear
        noise_std = noise_var ** 0.5
        noise = torch.randn_like(x) * noise_std
        return x + noise

    def forward(self, x, snr_db=None):
        if self.channel_type == "none" or snr_db is None:
            return x
        x = self.power_normalize(x)
        if self.channel_type == "awgn":
            return self.awgn(x, snr_db)
        if self.channel_type == "rayleigh":
            fading = torch.randn_like(x) / math.sqrt(2.0)
            y = x * fading
            return self.awgn(y, snr_db)
        return x


class MVSCNetCopy(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.cbr_weight = float(args.cbr_weight)

        embed_dim = config.mvsc_embed_dim
        latent_dim = config.mvsc_latent_dim
        crop_size = config.image_dims[1]
        num_views = config.mvsc_num_views

        self.individual_encoder = enc_mod.MVSC_Individual_Encoder(
            img_size=crop_size,
            patch_size=2,
            in_chans=3,
            embed_dim=embed_dim,
        )

        self.common_encoder = enc_mod.MVSC_Commonality_Encoder(
            dim=embed_dim,
            input_resolution=(crop_size // 8, crop_size // 8),
            depth=config.mvsc_common_depth,
            num_heads=config.mvsc_common_heads,
        )

        self.jscc_encoder = enc_mod.MVSC_JSCC_Encoder(
            dim=embed_dim,
            latent_dim=latent_dim,
        )

        self.channel = SimpleChannel(channel_type=args.channel_type)

        self.jscc_decoder = dec_mod.MVSC_JSCC_Decoder(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            compressed_num_views=max(1, num_views // 2),
        )

        self.common_decoder = dec_mod.MVSC_Commonality_Decoder(
            dim=embed_dim,
            input_resolution=(crop_size // 8, crop_size // 8),
            num_views=num_views,
            depth=config.mvsc_common_depth,
            num_heads=config.mvsc_common_heads,
        )

        self.individual_decoder = dec_mod.MVSC_Individual_Decoder(
            img_size=crop_size,
            patch_size=4,
            out_chans=3,
            embed_dim=embed_dim,
            input_resolution=(crop_size // 8, crop_size // 8),
        )

        self.distortion_fn = nn.MSELoss()

    def estimate_cbr(self, z, x):
        # Approximate CBR:
        # transmitted components * bits_per_component / input pixels
        bits_per_comp = float(getattr(self.args, "cbr_bits_per_component", 3.0))
        transmitted_components = z[0].numel()
        _, T, V, C, H, W = x.shape
        input_components = T * V * H * W * C
        cbr = (transmitted_components * bits_per_comp) / max(input_components, 1)
        return x.new_tensor(float(cbr))

    def forward(self, x, given_SNR=None):
        l = self.individual_encoder(x)
        s = self.common_encoder(l)
        z = self.jscc_encoder(s)

        used_snr = given_SNR
        if self.args.channel_type != "none" and used_snr is None:
            snr_values = [float(v.strip()) for v in str(self.args.multiple_snr).split(",") if v.strip()]
            used_snr = random.choice(snr_values) if snr_values else 15.0

        z_hat = self.channel(z, snr_db=used_snr)
        s_hat = self.jscc_decoder(z_hat)
        l_hat = self.common_decoder(s_hat)
        x_hat = self.individual_decoder(l_hat)

        distortion = self.distortion_fn(x_hat, x)
        cbr = self.estimate_cbr(z, x)
        total_loss = distortion + self.cbr_weight * cbr

        aux = {
            "distortion": distortion.detach(),
            "cbr": cbr.detach(),
        }
        return x_hat, used_snr, total_loss, aux


# =========================================================
# 6. Args / config
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train MVSC copy-model on full Harmony4D-style multi-view data")

    parser.add_argument("--train-root", type=str, default=DEFAULT_HARMONY_TRAIN_ROOT)
    parser.add_argument("--val-root", type=str, default=DEFAULT_HARMONY_VAL_ROOT)
    parser.add_argument("--output-dir", type=str, default="runs/mvsc_copy_full_dataset")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume-weights-only", action="store_true")
    parser.add_argument("--resume-ignore-mismatch", action="store_true")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step-unit", type=str, default="batch", choices=["batch", "epoch"])
    parser.add_argument("--lr-step-interval", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--resize-shorter-to", type=int, default=0)
    parser.add_argument("--train-repeat", type=int, default=2000)
    parser.add_argument("--val-repeat", type=int, default=100)
    parser.add_argument("--repeat-per-exo", action="store_true")
    parser.add_argument("--min-common-frames", type=int, default=8)

    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--common-depth", type=int, default=2)
    parser.add_argument("--common-heads", type=int, default=4)

    parser.add_argument("--channel-type", type=str, default="awgn", choices=["awgn", "rayleigh", "none"])
    parser.add_argument("--multiple-snr", type=str, default="15")
    parser.add_argument("--given-snr", type=float, default=15.0)
    parser.add_argument("--val-given-snr", type=float, default=15.0)
    parser.add_argument("--cbr-weight", type=float, default=0.0)
    parser.add_argument("--cbr-bits-per-component", type=float, default=3.0)

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1)

    return parser.parse_args()


def make_model_args(args):
    return SimpleNamespace(
        channel_type=args.channel_type,
        multiple_snr=args.multiple_snr,
        cbr_weight=args.cbr_weight,
        cbr_bits_per_component=float(args.cbr_bits_per_component),
        distortion_metric="MSE",
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


# =========================================================
# 7. Resume
# =========================================================
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

    incompatible = model.load_state_dict(model_state, strict=strict)
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

    if not (load_optimizer and isinstance(checkpoint, dict) and "optimizer" in checkpoint):
        print("[Info] Resume mode: weights only.")
        return start_epoch, global_step, optimizer_step, best_val_loss, best_val_psnr

    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except Exception as exc:
        print(f"[Warn] Failed to load optimizer state: {exc}")

    if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception as exc:
            print(f"[Warn] Failed to load scheduler state: {exc}")

    if scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        try:
            scaler.load_state_dict(checkpoint["scaler"])
        except Exception as exc:
            print(f"[Warn] Failed to load scaler state: {exc}")

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    optimizer_step = int(checkpoint.get("optimizer_step", global_step))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    best_val_psnr = float(checkpoint.get("best_val_psnr", float("-inf")))

    return start_epoch, global_step, optimizer_step, best_val_loss, best_val_psnr


# =========================================================
# 8. Train / eval
# =========================================================
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
):
    model.train()
    loss_meter = AverageMeter()
    distortion_meter = AverageMeter()
    cbr_meter = AverageMeter()
    psnr_meter = AverageMeter()

    start = time.time()
    last_log_time = start

    nonfinite_skip_count = 0
    accum_count = 0
    accum_steps = 1  # 兼容性保留，后续可支持梯度累积
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        global_step += 1
        x = batch["x"].to(device, non_blocking=True)

        if not torch.isfinite(x).all():
            nonfinite_skip_count += 1
            print(f"[Warn] Non-finite input detected at epoch={epoch} step={step}. skip_count={nonfinite_skip_count} {_batch_debug_desc(batch)}")
            if accum_count > 0:
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
            continue

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(args.amp, device):
            x_hat, used_snr, loss, aux = model(x, given_SNR=given_snr_override)

        loss_is_finite = bool(torch.isfinite(loss).all().item())
        x_hat_is_finite = bool(torch.isfinite(x_hat).all().item())
        if not (loss_is_finite and x_hat_is_finite):
            nonfinite_skip_count += 1
            loss_text = float(loss.detach().float().mean().item()) if torch.numel(loss) > 0 else float("nan")
            print(f"[Warn] Non-finite forward detected at epoch={epoch} step={step} loss={loss_text:.6f} snr={used_snr} skip_count={nonfinite_skip_count} {_batch_debug_desc(batch)}")
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
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
        distortion_meter.update(aux["distortion"].item(), bsz)
        cbr_meter.update(aux["cbr"].item(), bsz)
        psnr_meter.update(compute_psnr(x_hat.detach(), x.detach()), bsz)

        if step % args.log_interval == 0 or step == len(loader):
            now = time.time()
            batch_time = now - last_log_time
            last_log_time = now
            lr_now = optimizer.param_groups[0]["lr"]
            # 梯度统计
            grad_means = []
            grad_maxs = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad_means.append(p.grad.abs().mean().item())
                    grad_maxs.append(p.grad.abs().max().item())
            grad_mean = float(np.mean(grad_means)) if grad_means else 0.0
            grad_max = float(np.max(grad_maxs)) if grad_maxs else 0.0
            print(
                f"[Train] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss_meter.val:.6f} avg_loss={loss_meter.avg:.6f} "
                f"dist={distortion_meter.val:.6f} avg_dist={distortion_meter.avg:.6f} "
                f"cbr={cbr_meter.val:.6f} avg_cbr={cbr_meter.avg:.6f} "
                f"psnr={psnr_meter.val:.3f} avg_psnr={psnr_meter.avg:.3f} "
                f"snr={used_snr} lr={lr_now:.6e} time={batch_time:.1f}s "
                f"grad_mean={grad_mean:.2e} grad_max={grad_max:.2e}"
            )

    if nonfinite_skip_count > 0:
        print(f"[Warn] Epoch {epoch} skipped non-finite train batches: {nonfinite_skip_count}")

    return loss_meter.avg, psnr_meter.avg, distortion_meter.avg, cbr_meter.avg, global_step, optimizer_step


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
                print(f"[Warn] Non-finite input in validation at epoch={epoch} step={step}. skip_count={nonfinite_skip_count} {_batch_debug_desc(batch)}")
                continue

            x_hat, used_snr, loss, aux = model(x, given_SNR=given_snr_override)
            loss_is_finite = bool(torch.isfinite(loss).all().item())
            x_hat_is_finite = bool(torch.isfinite(x_hat).all().item())
            if not (loss_is_finite and x_hat_is_finite):
                nonfinite_skip_count += 1
                loss_text = float(loss.detach().float().mean().item()) if torch.numel(loss) > 0 else float("nan")
                print(f"[Warn] Non-finite validation forward at epoch={epoch} step={step} loss={loss_text:.6f} snr={used_snr} skip_count={nonfinite_skip_count} {_batch_debug_desc(batch)}")
                continue

            bsz = x.shape[0]
            loss_meter.update(loss.item(), bsz)
            distortion_meter.update(aux["distortion"].item(), bsz)
            cbr_meter.update(aux["cbr"].item(), bsz)
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
        raise RuntimeError(f"Validation produced no finite batches at epoch={epoch}. skipped={nonfinite_skip_count}")

    if nonfinite_skip_count > 0:
        print(f"[Warn] Epoch {epoch} skipped non-finite val batches: {nonfinite_skip_count}")

    if loss_meter.count == 0:
        raise RuntimeError(f"Validation produced no finite batches at epoch={epoch}.")

    return loss_meter.avg, psnr_meter.avg, distortion_meter.avg, cbr_meter.avg


# =========================================================
# 9. Main
# =========================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    if device.type != "cuda" and args.channel_type in {"awgn", "rayleigh"}:
        print(f"[Warn] CUDA unavailable. Fallback channel_type: {args.channel_type} -> none")
        args.channel_type = "none"

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train_dataset = build_dataset(args.train_root, args, is_train=True)
    val_dataset = build_dataset(args.val_root, args, is_train=False)

    train_loader = build_loader(train_dataset, args, is_train=True, device=device)
    val_loader = build_loader(val_dataset, args, is_train=False, device=device)

    model_args = make_model_args(args)
    config = make_config(args, device)
    model = MVSCNetCopy(model_args, config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_step_unit == "batch":
        scheduler_steps_per_epoch = max(1, math.ceil(len(train_loader) / args.lr_step_interval))
        scheduler_t_max = max(1, args.epochs * scheduler_steps_per_epoch)
    else:
        scheduler_t_max = max(1, math.ceil(args.epochs / args.lr_step_interval))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max)

    scaler = make_grad_scaler(args.amp, device)

    start_epoch = 1
    global_step = 0
    optimizer_step = 0
    best_val_loss = float("inf")
    best_val_psnr = float("-inf")

    if args.resume:
        start_epoch, global_step, optimizer_step, best_val_loss, best_val_psnr = load_resume_state(
            resume_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            output_dir=args.output_dir,
            load_optimizer=not args.resume_weights_only,
            strict=not args.resume_ignore_mismatch,
        )

    print(f"Device: {device}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Using encoder copy: {ENCODER_COPY_PATH}")
    print(f"Using decoder copy: {DECODER_COPY_PATH}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(
        f"[Info] LR scheduler: cosine, unit={args.lr_step_unit}, "
        f"interval={args.lr_step_interval}, T_max={scheduler_t_max}"
    )

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
            "args": vars(args),
        }

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_psnr, train_dist, train_cbr, global_step, optimizer_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
            given_snr_override=args.given_snr,
            global_step=global_step,
            optimizer_step=optimizer_step,
        )

        val_loss, val_psnr, val_dist, val_cbr = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            args=args,
            epoch=epoch,
            given_snr_override=args.val_given_snr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = build_state(epoch, global_step)
            state["best_val_loss"] = best_val_loss
            save_checkpoint(state, os.path.join(args.output_dir, "best.pt"))
            print(f"Saved new best checkpoint with val_loss={best_val_loss:.6f}")

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            state = build_state(epoch, global_step)
            state["best_val_psnr"] = best_val_psnr
            save_checkpoint(state, os.path.join(args.output_dir, "best_psnr.pt"))
            print(f"Saved new best-psnr checkpoint with val_psnr={best_val_psnr:.3f}")

        if args.lr_step_unit == "epoch" and epoch % args.lr_step_interval == 0:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.6f} train_dist={train_dist:.6f} train_cbr={train_cbr:.6f} train_psnr={train_psnr:.3f} "
            f"val_loss={val_loss:.6f} val_dist={val_dist:.6f} val_cbr={val_cbr:.6f} val_psnr={val_psnr:.3f} "
            f"lr={lr_now:.6e} opt_steps={optimizer_step}"
        )

        state = build_state(epoch, global_step)
        save_checkpoint(state, os.path.join(args.output_dir, "latest.pt"))

        if epoch % args.save_every == 0:
            save_checkpoint(state, os.path.join(args.output_dir, f"epoch_{epoch:03d}.pt"))

    print("Training finished.")


if __name__ == "__main__":
    main()