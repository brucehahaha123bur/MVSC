import os
import random
from typing import List, Dict, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_image_file(x: str) -> bool:
    return x.lower().endswith(IMG_EXTS)


def natural_sort_key(name: str):
    stem = os.path.splitext(name)[0]
    try:
        return int(stem)
    except ValueError:
        return stem


class Harmony4DMVSCDataset(Dataset):
    """
    用于 MVSC 风格训练的 Harmony4D exo 多视角 GOP 数据集

    目录结构（你的当前结构）：
        exo/
            cam01/
                images/
                    00000.jpg
                    00001.jpg
                    ...
            cam02/
                images/
                    ...
            ...
            cam22/
                images/
                    ...

    输出：
        x: torch.FloatTensor, shape = [T, V, 3, crop_size, crop_size], 值域 [0,1]

    说明：
    1. 每次随机选 V 个 camera
    2. 在公共帧编号里随机选一个连续起点，取 T 帧
    3. 对所有 T×V 图像做同一个随机裁剪，保证时空对齐
    """

    def __init__(
        self,
        root: str,
        num_views: int = 4,
        num_frames: int = 4,
        crop_size: int = 256,
        resize_shorter_to: int = None,
        random_crop: bool = True,
        random_flip: bool = True,
        min_common_frames: int = 8,
        repeat: int = 1000,
    ):
        super().__init__()
        self.root = root
        self.num_views = num_views
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.resize_shorter_to = resize_shorter_to
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.min_common_frames = min_common_frames
        self.repeat = repeat

        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"root 不存在: {self.root}")

        self.cam_dirs = self._find_camera_dirs(self.root)
        if len(self.cam_dirs) < self.num_views:
            raise ValueError(
                f"可用相机数不足: found={len(self.cam_dirs)}, need={self.num_views}"
            )

        self.cam_frames: Dict[str, List[str]] = {}
        self.cam_frame_sets: Dict[str, set] = {}

        for cam_name, img_dir in self.cam_dirs:
            frames = [
                f for f in os.listdir(img_dir)
                if is_image_file(f)
            ]
            frames = sorted(frames, key=natural_sort_key)
            if len(frames) == 0:
                continue
            self.cam_frames[cam_name] = frames
            self.cam_frame_sets[cam_name] = set(frames)

        self.valid_cams = sorted(self.cam_frames.keys())
        if len(self.valid_cams) < self.num_views:
            raise ValueError(
                f"有效相机数不足: valid={len(self.valid_cams)}, need={self.num_views}"
            )

        # 预构建一些可采样 camera 组合，减少 __getitem__ 时反复试错
        self.valid_groups = self._build_valid_camera_groups(max_trials=5000)
        if len(self.valid_groups) == 0:
            raise RuntimeError("没有找到满足公共连续帧要求的 camera 组合，请检查数据结构。")

        print(f"[Harmony4D] root = {self.root}")
        print(f"[Harmony4D] valid cams = {len(self.valid_cams)}")
        print(f"[Harmony4D] valid groups = {len(self.valid_groups)}")

    def _find_camera_dirs(self, root: str) -> List[Tuple[str, str]]:
        """
        返回 [(cam_name, img_dir), ...]
        支持两种：
            exo/cam01/images/*.jpg
            exo/cam01/*.jpg
        """
        out = []
        for name in sorted(os.listdir(root)):
            cam_path = os.path.join(root, name)
            if not os.path.isdir(cam_path):
                continue
            if not name.startswith("cam"):
                continue

            img_dir = os.path.join(cam_path, "images")
            if os.path.isdir(img_dir):
                out.append((name, img_dir))
            else:
                out.append((name, cam_path))
        return out

    def _build_valid_camera_groups(self, max_trials: int = 5000) -> List[List[str]]:
        """
        随机采样 camera 组合，筛出存在足够公共连续帧的组合
        """
        groups = []
        seen = set()

        cams = self.valid_cams[:]
        trials = 0
        target_groups = min(256, max(16, len(cams) * 8))

        while trials < max_trials and len(groups) < target_groups:
            trials += 1
            chosen = tuple(sorted(random.sample(cams, self.num_views)))
            if chosen in seen:
                continue
            seen.add(chosen)

            common = self._common_frames_of_cams(list(chosen))
            if len(common) < max(self.min_common_frames, self.num_frames):
                continue

            runs = self._find_consecutive_runs(common)
            ok = any(len(run) >= self.num_frames for run in runs)
            if ok:
                groups.append(list(chosen))

        return groups

    def _common_frames_of_cams(self, cam_names: List[str]) -> List[str]:
        common = None
        for cam in cam_names:
            s = self.cam_frame_sets[cam]
            common = s if common is None else (common & s)
        if common is None:
            return []
        return sorted(list(common), key=natural_sort_key)

    def _find_consecutive_runs(self, frame_names: List[str]) -> List[List[str]]:
        """
        例如 ['00001.jpg','00002.jpg','00003.jpg','00005.jpg']
        -> [['00001.jpg','00002.jpg','00003.jpg'], ['00005.jpg']]
        """
        if len(frame_names) == 0:
            return []

        nums = [int(os.path.splitext(x)[0]) for x in frame_names]
        runs = []
        cur = [frame_names[0]]

        for i in range(1, len(frame_names)):
            if nums[i] == nums[i - 1] + 1:
                cur.append(frame_names[i])
            else:
                runs.append(cur)
                cur = [frame_names[i]]
        runs.append(cur)
        return runs

    def _sample_views_and_frames(self) -> Tuple[List[str], List[str]]:
        cam_names = random.choice(self.valid_groups)
        common = self._common_frames_of_cams(cam_names)
        runs = self._find_consecutive_runs(common)
        valid_runs = [r for r in runs if len(r) >= self.num_frames]
        if len(valid_runs) == 0:
            raise RuntimeError("理论上不该发生：选中的 camera group 没有足够连续帧。")

        run = random.choice(valid_runs)
        start = random.randint(0, len(run) - self.num_frames)
        frame_names = run[start:start + self.num_frames]
        return cam_names, frame_names

    def _load_image(self, cam_name: str, frame_name: str) -> Image.Image:
        img_dir = None
        for n, d in self.cam_dirs:
            if n == cam_name:
                img_dir = d
                break
        if img_dir is None:
            raise FileNotFoundError(f"找不到相机目录: {cam_name}")

        path = os.path.join(img_dir, frame_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到图像: {path}")

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            from PIL import UnidentifiedImageError
            if isinstance(e, UnidentifiedImageError) or isinstance(e, OSError):
                print(f"[Warn] PIL.UnidentifiedImageError/OSError: {path}, err={e}")
                return None
            else:
                raise
        return img

    def _maybe_resize(self, img: Image.Image) -> Image.Image:
        if self.resize_shorter_to is None:
            return img

        w, h = img.size
        shorter = min(w, h)
        if shorter == self.resize_shorter_to:
            return img

        scale = self.resize_shorter_to / shorter
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return img.resize((new_w, new_h), resample=Image.BICUBIC)

    def _get_crop_params(self, w: int, h: int):
        th, tw = self.crop_size, self.crop_size

        if h < th or w < tw:
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)
            return 0, 0, pad_h, pad_w, 0, 0

        if self.random_crop:
            top = random.randint(0, h - th)
            left = random.randint(0, w - tw)
        else:
            top = (h - th) // 2
            left = (w - tw) // 2

        return top, left, 0, 0, th, tw

    def __len__(self):
        return self.repeat

    def __getitem__(self, index):
        cam_names, frame_names = self._sample_views_and_frames()

        # 先读第一张，确定统一 crop 参数
        first_img = self._load_image(cam_names[0], frame_names[0])
        if first_img is None:
            return None
        first_img = self._maybe_resize(first_img)

        w, h = first_img.size
        top, left, pad_h, pad_w, th, tw = self._get_crop_params(w, h)
        do_flip = self.random_flip and (random.random() < 0.5)

        frames_out = []
        for t in range(self.num_frames):
            views_out = []
            for v in range(self.num_views):
                img = self._load_image(cam_names[v], frame_names[t])
                if img is None:
                    # 只要有一张图片坏掉，整个样本直接跳过
                    return None
                img = self._maybe_resize(img)

                if pad_h > 0 or pad_w > 0:
                    img = TF.pad(img, padding=[0, 0, pad_w, pad_h], fill=0)

                img = TF.crop(img, top=top, left=left, height=self.crop_size, width=self.crop_size)

                if do_flip:
                    img = TF.hflip(img)

                x = TF.to_tensor(img)  # [3,H,W], [0,1]
                views_out.append(x)

            # 只要有一张视角图片坏掉，整个帧直接跳过
            if len(views_out) != self.num_views:
                return None
            views_out = torch.stack(views_out, dim=0)   # [V,3,H,W]
            frames_out.append(views_out)

        # 只要有一帧坏掉，整个样本直接跳过
        if len(frames_out) != self.num_frames:
            return None
        x = torch.stack(frames_out, dim=0)  # [T,V,3,H,W]

        sample = {
            "x": x,
            "cam_names": cam_names,
            "frame_names": frame_names,
        }
        return sample