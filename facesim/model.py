from torch import nn
import torch


class FaceSim(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1),
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
    model = FaceSim()
    
    eyebrow = torch.randn(1, 3, 32, 64)
    eye = torch.randn(1, 3, 32, 48)
    nose = torch.randn(1, 3, 64, 32)
    lips = torch.randn(1, 3, 32, 64)

    for part in [eyebrow, eye, nose, lips]:
        print(model(part).shape)