from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import clearml

from lightning_system import LightningModel
from model import GarmentClassifier

task = clearml.Task.init(
    project_name="fashion-mnist-classification", task_name="clothes classification"
)
dataset_id = "9ba94ca4389342e0a4e965a50695e472"

# Also supports dataset name
clearml_dataset = clearml.Dataset.get(dataset_id)
dataset_location = clearml_dataset.get_local_copy(use_soft_links=True)
training_set = ImageFolder(f"{dataset_location}/train", transform=transforms.ToTensor())
validation_set = ImageFolder(
    f"{dataset_location}/valid", transform=transforms.ToTensor()
)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = DataLoader(training_set, batch_size=40, shuffle=True, num_workers=4)
validation_loader = DataLoader(
    validation_set, batch_size=40, shuffle=False, num_workers=4
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = LightningModel(GarmentClassifier()).to(device)
logger = TensorBoardLogger("logs")

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=[0] if torch.cuda.is_available() else "auto",
    max_epochs=1000,
    logger=logger,
)
trainer.fit(
    model=net, train_dataloaders=training_loader, val_dataloaders=validation_loader
)
