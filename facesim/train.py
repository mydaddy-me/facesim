import pytorch_lightning as pl
import torch
import torch.nn.functional as F

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

        ap_sim = F.cosine_similarity(ae, pe)
        an_sim = F.cosine_similarity(ae, ne)

        sim = torch.hstack([ap_sim, an_sim])

        assert sim.shape == (
            len(batch), 2), f"Expected (batch, 2) got {sim.shape}"

        prob = F.softmax(sim, dim=1)
        pos_prob = prob[:, 0]

        return F.cross_entropy(
            pos_prob,
            torch.ones_like(pos_prob))

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
