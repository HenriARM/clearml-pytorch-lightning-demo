import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchmetrics import Accuracy

import pytorch_lightning as pl
from clearml import Logger
import random
from torchvision.utils import make_grid


class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LightningModel(pl.LightningModule):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.backbone(x.float())
        loss = self.criterion(y_pred, y)

        accuracy = Accuracy("multiclass", num_classes=10)
        acc = accuracy(y_pred, y)

        self.log("train_accuracy", acc, prog_bar=True)
        self.log("train_loss", loss)

        # Assuming x is a tensor of size [batch_size, channels, height, width]
        grid_image = make_grid(x, normalize=True)

        # Convert the grid image tensor to a NumPy array
        grid_np = grid_image.permute(1, 2, 0).mul(255).byte().cpu().numpy()

        clearml_logger = Logger.current_logger()

        clearml_logger.report_scalar("My custom scalar", "random", random.random(), batch_idx)

        # Send the grid image to ClearML
        clearml_logger.report_image(
            "input_grid", f"image_{batch_idx}", iteration=self.current_epoch, image=grid_np
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.backbone(x.float())
        loss = self.criterion(y_pred, y)

        accuracy = Accuracy("multiclass", num_classes=10)
        acc = accuracy(y_pred, y)

        self.log("val_accuracy", acc, prog_bar=True)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.backbone.parameters(), lr=1e-3)
        return optimizer