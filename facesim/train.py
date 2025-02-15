from typing import Literal

import lightning.pytorch as pl
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.nn.functional import cosine_similarity as cos
from torch.nn.functional import cross_entropy, softmax
from torchmetrics.functional import accuracy

from facesim.model import FaceSim, Parts


class FaceSimModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def step(self, batch, name: Literal['train', 'val']):
        a, p, n = batch
        ae, pe, ne = self.model(a), self.model(p), self.model(n)

        ap_sim = cos(ae, pe)
        an_sim = cos(ae, ne)

        sim = torch.stack([ap_sim, an_sim], dim=1)

        assert sim.shape == (len(a), 2), \
            f"Expected ({len(a)}, 2) got {sim.shape}"

        prob = softmax(sim, dim=1)
        self.log(
            f'acc/{name}',
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

        self.log(f'loss/{name}', loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4)


if __name__ == "__main__":
    seed_everything(0)

    parts = Parts()

    model = FaceSim()
    module = FaceSimModule(model)

    trainer = pl.Trainer(
        max_epochs=100,
        deterministic=True,
        callbacks=[

            ModelCheckpoint(
                monitor='loss/val',
                mode='min'),

            # EarlyStopping(
            #     monitor='loss/val',
            #     patience=3,
            #     mode='min')
        ])

    trainer.fit(
        module,
        train_dataloaders=parts.dataloader(2**6),
        val_dataloaders=parts.dataloader(2**6))
