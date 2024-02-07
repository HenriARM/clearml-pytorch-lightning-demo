import torchvision.transforms as transforms
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader
import tqdm
import shutil

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

training_set = torchvision.datasets.FashionMNIST(
    "./fashion_raw", transform=transform, train=True, download=True
)
validation_set = torchvision.datasets.FashionMNIST(
    "./fashion_raw", transform=transform, train=False, download=True
)

bs = 10
training_loader = DataLoader(training_set, batch_size=bs, shuffle=False)
validation_loader = DataLoader(validation_set, batch_size=bs, shuffle=False)

dataset_dir = Path("./dataset")
dataset_dir.mkdir(exist_ok=True)


def process_class_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


image_counts = {
    split: {process_class_name(class_name): 0 for class_name in training_set.classes}
    for split in ["train", "valid"]
}

# Convert FashionMNIST dataset to png images by iterating through the dataset
# and saving each image as a png file
for dl, ds, split in [
    [training_loader, training_set, "train"],
    [validation_loader, validation_set, "valid"],
]:
    for name in ds.classes:
        # replace class names with underscores
        class_name = name.replace("/", "_").replace(" ", "_")
        class_path = dataset_dir / split / class_name
        class_path.mkdir(exist_ok=True, parents=True)

    # loop batches
    with tqdm.tqdm(total=len(ds)) as pbar:
        for image, class_idx in dl:
            for image_idx, image_tensor in enumerate(image):
                class_name = process_class_name(ds.classes[class_idx[image_idx]])
                class_path = dataset_dir / split / class_name
                current_image_count = image_counts[split][class_name]
                torchvision.utils.save_image(
                    image_tensor,
                    class_path / f"{current_image_count}.png",
                )
                # Increment the image count for the class
                image_counts[split][class_name] += 1
                pbar.update(1)

# Remove the fashion_raw directory and all its contents
try:
    shutil.rmtree("./fashion_raw")
    print("fashion_raw directory has been removed successfully.")
except Exception as e:
    print(f"Error removing directory ./fashion_raw: {e}")
