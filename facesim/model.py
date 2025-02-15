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

        anchor = choice(imgs(l1))
        positive = choice(imgs(l1))
        negative = choice(imgs(l2))

        return anchor, positive, negative


class FaceSim(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 2, 1, bias=False),
                nn.BatchNorm2d(o),
                nn.Sigmoid())

        self.net = nn.Sequential(
            # eyebrow   32x64
            # eye       32x48
            # nose      64x32
            # lips      32x64

            blk(3, 8),     # 64
            blk(8, 16),    # 32
            blk(16, 32),   # 16
            blk(32, 64),   # 8
            nn.Conv2d(64, 128, 4, 1, 0))

    def forward(self, x):
        h, w = x.shape[1:]
        pl = (64 - w) // 2
        pr = 64 - w - pl
        pt = (64 - h) // 2
        pb = 64 - h - pt
        x = torch.nn.functional.pad(x, (pl, pr, pt, pb))

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
