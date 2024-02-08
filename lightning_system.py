import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
import pytorch_lightning as pl
from clearml import Logger
from torchvision.utils import make_grid


class LightningModel(pl.LightningModule):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"],)
        self.backbone = backbone
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy("multiclass", num_classes=10).to(self.device)

    def shared_step(self, batch, batch_idx, step_name):
        x, y = batch
        y_pred = self.backbone(x.float())
        loss = self.criterion(y_pred, y)
        self.log(f"{step_name}_loss", loss)
        self.accuracy(y_pred, y)
        acc = self.accuracy.compute()
        self.log(f"{step_name}_accuracy", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, "train")
        # Assuming x is a tensor of size [batch_size, channels, height, width]
        grid_image = make_grid(batch[0], normalize=True)
        # Convert the grid image tensor to a NumPy array
        grid_np = grid_image.permute(1, 2, 0).mul(255).byte().cpu().numpy()
        clearml_logger = Logger.current_logger()
        clearml_logger.report_image(
            "input_grid",
            f"image_{batch_idx}",
            iteration=self.current_epoch,
            image=grid_np,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.backbone.parameters(), lr=1e-3)
        return optimizer
