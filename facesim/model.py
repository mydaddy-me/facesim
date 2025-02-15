from torch import nn

class FaceSim(nn.Model):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            # eyebrow   24x48
            # eye       32x24
            # nose      48x24 
            # lips      16x48 
        )