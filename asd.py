import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from pytorch_lightning.loggers import TensorBoardLogger
from clearml import Task


class LitClassifier(pl.LightningModule):
    def __init__(self, hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == '__main__':
    pl.seed_everything(0)

    task: Task = Task.init(project_name="examples", task_name="pytorch lightning MNIST")

    # ------------
    # data
    # ------------
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=40)
    val_loader = DataLoader(mnist_val, batch_size=40)
    test_loader = DataLoader(mnist_test, batch_size=40)

    logger = TensorBoardLogger("logs")

    # ------------
    # model
    # ------------
    model = LitClassifier(128, 0.0001)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(max_epochs=2, logger=logger)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(dataloaders=test_loader)
    task.flush()