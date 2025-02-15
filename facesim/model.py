from collections import defaultdict
from functools import cache
from random import choice

import torch
from PIL import Image
from torch import nn
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from facesim.fs import fs


def pad64x64(img):
    h, w = img.shape[1:]
    return torch.nn.functional.pad(img, (0, 64 - w, 0, 64 - h))


class Parts:
    def __init__(self, parts=[fs.eyebrow, fs.eye, fs.nose, fs.lips]):
        self.parts = parts
        self.imgs = defaultdict(lambda: defaultdict(list))

        for part in self.parts:
            for f in tqdm(part.rglob('*.jpg')):
                label = f.parent.name
                part = f.parent.parent.name
                self.imgs[part][label].append(f)

    def dataset(self, length: int):
        return PartsDataset(
            imgs=self.imgs,
            length=length,
            parts=self.parts)

    def dataloader(self, len: int, bs: int = 32):
        return torch.utils.data.DataLoader(
            self.dataset(len),
            batch_size=bs,
            num_workers=4,
            shuffle=False)


class PartsDataset(torch.utils.data.Dataset):
    def __init__(self, *, imgs, length, parts):
        self.imgs = imgs
        self.parts = parts
        self.length = length

    def __len__(self):
        return self.length

    @cache
    def __getitem__(self, idx):
        part = choice(self.parts).name
        labels = list(self.imgs[part].keys())

        l1 = choice(labels)
        l2 = choice(labels)

        def get_img(label):
            img = choice(self.imgs[part][label])
            img = Image.open(img)
            img = to_tensor(img)
            img = pad64x64(img)

            return img

        anchor = get_img(l1)
        positive = get_img(l1)
        negative = get_img(l2)

        return anchor, positive, negative


class FaceSim(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 2, 1, bias=False),
                nn.BatchNorm2d(o),
                nn.ReLU())

        def io(ios: list[int]):
            return zip(ios[:-1], ios[1:])

        self.net = nn.Sequential(
            # eyebrow   32x64
            # eye       32x48
            # nose      64x32
            # lips      32x64

            *[blk(i, o) for i, o in io([3, 8, 16, 32, 64])],
            nn.Conv2d(64, 128, 4, 1, 0))

    def forward(self, x):
        assert x.shape[1:] == (3, 64, 64), \
            f"Expected (3, 64, 64) got {x.shape[1:]}"

        embd = self.net(x).squeeze()
        assert embd.shape == (len(x), 128), \
            f"Expected ({len(x)}, 128,) got {embd.shape}"

        return embd


if __name__ == "__main__":

    model = FaceSim()
    e = model(torch.rand(32, 3, 64, 64))
    print(e.shape)
