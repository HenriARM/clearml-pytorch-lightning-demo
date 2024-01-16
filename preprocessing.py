import torchvision.transforms as transforms
import torchvision

from pathlib import Path
from torch.utils.data import DataLoader
import tqdm
import concurrent.futures

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

training_set = torchvision.datasets.FashionMNIST(
    "./fashion_raw", transform=transform, train=True, download=True
)
validation_set = torchvision.datasets.FashionMNIST(
    "./fashion_raw", transform=transform, train=False, download=True
)

bs = 100
training_loader = DataLoader(training_set, batch_size=bs, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=bs, shuffle=False)

dataset_dir = Path("./dataset")
dataset_dir.mkdir(exist_ok=True)

# Convert FashionMNIST dataset to png images by iteratin through the dataset
# and saving each image as a png file
for dl, ds, split in [
    [training_loader, training_set, "train"],
    [validation_loader, validation_set, "valid"],
]:
    num_batches_per_epoch = 10

    for name in ds.classes:
        class_path = dataset_dir / split / name.split("/")[0]
        class_path.mkdir(exist_ok=True, parents=True)

    with tqdm.tqdm(total=len(ds)) as pbar:
        for idx, (image, class_idx) in enumerate(dl):
            # Stop after one epoch
            if idx >= num_batches_per_epoch:
                break

            # Create a thread pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Iterate batch
                for image_idx, image_tensor in enumerate(image):
                    class_path = (
                        dataset_dir
                        / split
                        / ds.classes[class_idx[image_idx]].split("/")[0]
                    )

                    executor.submit(
                        torchvision.utils.save_image,
                        image_tensor,
                        class_path / f"{image_idx}.png",
                    )
