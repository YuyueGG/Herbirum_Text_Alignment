from __future__ import annotations

from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM


class ResizeShortSideAndPad:
    def __init__(self, size: int = 224, fill: int = 128):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        if width < height:
            new_width, new_height = self.size, int(height * (self.size / width))
        else:
            new_height, new_width = self.size, int(width * (self.size / height))

        image = image.resize((new_width, new_height), Image.BICUBIC)

        pad_left = max(0, (self.size - new_width) // 2)
        pad_right = max(0, self.size - new_width - pad_left)
        pad_top = max(0, (self.size - new_height) // 2)
        pad_bottom = max(0, self.size - new_height - pad_top)

        if any([pad_left, pad_right, pad_top, pad_bottom]):
            image = ImageOps.expand(image, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)

        if image.size != (self.size, self.size):
            image = image.crop((0, 0, self.size, self.size))

        return image


def build_transforms(augment: bool):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if augment:
        return T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1), interpolation=IM.BICUBIC),
                T.RandomHorizontalFlip(0.5),
                T.RandomApply([T.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.5),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )

    return T.Compose(
        [
            ResizeShortSideAndPad(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
