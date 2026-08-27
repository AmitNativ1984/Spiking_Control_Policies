import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from vae_depth.collision import cache_dir_for, cache_path_for


class DepthImageDataset(Dataset):
    """16-bit PNG depth images, paired with their collision-encoding target.

    The encoder input and the decoder target are DIFFERENT images: the input is raw depth,
    the target is the depth at which the drone's collision sphere first touches something
    (vae_depth/collision.py). That asymmetry is what forces the latent to carry collision
    information rather than a compressed depth map.

    The target is precomputed (vae_depth/precompute_collision.py) and loaded here, because
    building it per step costs about twice a full training step. Augmentation is sampled
    once and applied to both images, which is valid because the dilation commutes with
    crop/flip -- see tests/test_collision.py.

    In "legacy" mode the target is the input and train.py applies the old single-range
    min-pool instead; that path exists only to A/B the two encodings.
    """

    def __init__(self, image_paths: list, config, augment: bool = True):
        self.image_paths = image_paths
        self.config = config
        self.augment = augment
        self.legacy = config.collision_target_mode == "legacy"

        if not self.legacy:
            self.cache_dir = cache_dir_for(config)
            missing = [p for p in image_paths[:64]
                       if not os.path.exists(cache_path_for(p, config, self.cache_dir))]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} of the first 64 images have no cached collision target "
                    f"in {self.cache_dir}.\nBuild it with:\n"
                    f"    python -m vae_depth.precompute_collision\n"
                    f"or set collision_target_mode='legacy' to train the old encoding.")

    def __len__(self):
        return len(self.image_paths)

    def _load(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth = img.astype(np.float32) / 65535.0 * self.config.depth_scale
        return cv2.resize(
            depth,
            (self.config.target_width, self.config.target_height),
            interpolation=cv2.INTER_NEAREST,
        )

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        depth = self._load(path)
        target = depth if self.legacy else self._load(
            cache_path_for(path, self.config, self.cache_dir))

        # One sample of the augmentation parameters, applied to BOTH images. Sampling them
        # twice would pair an input with a target from a different view -- silently, since
        # both remain valid-looking depth maps.
        if self.augment and random.random() < self.config.crop_prob:
            scale = random.uniform(self.config.crop_scale_min, self.config.crop_scale_max)
            crop_h = int(self.config.target_height * scale)
            crop_w = int(self.config.target_width * scale)
            top = random.randint(0, self.config.target_height - crop_h)
            left = random.randint(0, self.config.target_width - crop_w)
            depth, target = (self._crop(a, top, left, crop_h, crop_w) for a in (depth, target))

        if self.augment and random.random() < self.config.flip_prob:
            depth = np.flip(depth, axis=1).copy()
            target = np.flip(target, axis=1).copy()

        return torch.from_numpy(depth).unsqueeze(0), torch.from_numpy(target).unsqueeze(0)

    def _crop(self, arr, top, left, crop_h, crop_w):
        arr = arr[top:top + crop_h, left:left + crop_w]
        return cv2.resize(
            arr,
            (self.config.target_width, self.config.target_height),
            interpolation=cv2.INTER_NEAREST,
        )
