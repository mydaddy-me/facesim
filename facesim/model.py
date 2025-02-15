from collections import defaultdict
from random import choice

import torch
from PIL import Image
from torch import nn
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from facesim.fs import fs


class PartsDataset(torch.utils.data.Dataset):
    def __init__(self, parts=[fs.eyebrow, fs.eye, fs.nose, fs.lips], length=2**16):
        self.length = length
        self.parts = parts
        self.imgs = defaultdict(lambda: defaultdict(list))

        for part in self.parts:
            for f in tqdm(part.rglob('*.jpg')):
                label = f.parent.name
                part = f.parent.parent.name
                img = Image.open(f)
                img = to_tensor(img)
                self.imgs[part][label].append(img)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        part = choice(self.parts).name
        labels = list(self.imgs[part].keys())

        l1 = choice(labels)
        l2 = choice(labels)

        def imgs(label):
            return self.imgs[part][label]

        def pad64x64(img):
            h, w = img.shape[1:]
            pad_h = (64 - h % 64) % 64
            pad_w = (64 - w % 64) % 64

            return torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))

        anchor = choice(imgs(l1))
        positive = choice(imgs(l1))
        negative = choice(imgs(l2))

        return (
            pad64x64(anchor),
            pad64x64(positive),
            pad64x64(negative))


class FaceSim(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1, bias=False),
                nn.BatchNorm2d(o),
                nn.Sigmoid())

        self.net = nn.Sequential(
            # eyebrow   32x64
            # eye       32x48
            # nose      64x32
            # lips      32x64

            blk(3, 8),     # 32
            blk(8, 16),    # 16
            blk(16, 32),   # 8
            blk(32, 64),   # 4
            blk(64, 128),  # 2
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        return self.net(x).squeeze()


if __name__ == "__main__":

    dataset = PartsDataset()
    model = FaceSim()

    for anchor, positive, negative in dataset:
        # anchor = model(anchor)
        # positive = model(positive)
        # negative = model(negative)

        print(anchor.shape, positive.shape, negative.shape)
        break
