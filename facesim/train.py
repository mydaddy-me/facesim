import pytorch_lightning as pl
import torch
from torch.nn.functional import cosine_similarity as cos
from torch.nn.functional import cross_entropy, softmax
from torchmetrics.functional import accuracy

from facesim.model import FaceSim, PartsDataset


class FaceSimModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        a, p, n = batch
        ae, pe, ne = self.model(a), self.model(p), self.model(n)

        ap_sim = cos(ae, pe)
        an_sim = cos(ae, ne)

        sim = torch.stack([ap_sim, an_sim], dim=1)

        assert sim.shape == (len(a), 2), \
            f"Expected ({len(a)}, 2) got {sim.shape}"

        prob = softmax(sim, dim=1)
        self.log(
            'acc',
            accuracy(
                prob[:, 0],
                torch.ones_like(ap_sim),
                task='binary'),
            prog_bar=True)

        loss = cross_entropy(
            sim,
            torch.zeros_like(
                ap_sim,
                dtype=torch.long))

        self.log('loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3)


if __name__ == "__main__":
    ds = PartsDataset()
    a, p, n = ds[0]

    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=32,
        num_workers=4,
        shuffle=False)

    model = FaceSim()
    module = FaceSimModule(model)

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(module, dl)
