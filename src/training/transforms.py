"""Image transforms shared by every family: pad to square, resize, train-time augmentation."""

from __future__ import annotations

from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PadToSquare:
    """Pad a PIL image with a constant colour so it is square, keeping the image centred."""

    def __init__(self, fill: Tuple[int, int, int] = (0, 0, 0)) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height:
            return image
        side = max(width, height)
        canvas = Image.new("RGB", (side, side), self.fill)
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        return canvas


class AddGaussianNoise:
    """Add Gaussian noise to a [0, 1] tensor with probability ``p``."""

    def __init__(self, std: float = 0.02, p: float = 0.3) -> None:
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return tensor
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)


def build_transforms(image_size: int, train: bool) -> T.Compose:
    """Eval: pad, resize, normalize. Train adds the proposal's augmentations (no hue/saturation)."""
    steps = [PadToSquare(), T.Resize((image_size, image_size))]
    if train:
        steps += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(1.0, 1.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    steps.append(T.ToTensor())
    if train:
        steps.append(AddGaussianNoise())
    steps.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return T.Compose(steps)
