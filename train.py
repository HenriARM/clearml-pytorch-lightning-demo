
from torchvision.datasets import ImageFolder
import clearml
from torch.utils.data import  DataLoader
import pytorch_lightning as pl
import torchvision.transforms as transforms
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_system import LightningModel, GarmentClassifier

task = clearml.Task.init(project_name="classification", task_name="garments")

dataset_id = "ee8ecc7f37e7403aa677f3e0859b1ee6"

# Also supports dataset name
clearml_dataset = clearml.Dataset.get(dataset_id)

dataset_location = clearml_dataset.get_local_copy(use_soft_links=True)

training_set = ImageFolder(f"{dataset_location}/train", transform=transforms.ToTensor())
validation_set = ImageFolder(
    f"{dataset_location}/valid", transform=transforms.ToTensor()
)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = DataLoader(training_set, batch_size=40, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=40, shuffle=False)

net = LightningModel(GarmentClassifier())

logger = TensorBoardLogger("logs")

trainer = pl.Trainer(
    max_epochs=1000,
    logger=logger
)
trainer.fit(
    model=net, train_dataloaders=training_loader, val_dataloaders=validation_loader
)
