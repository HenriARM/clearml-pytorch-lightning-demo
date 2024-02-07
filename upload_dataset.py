import clearml
from pathlib import Path

dataset = clearml.Dataset.create(
    dataset_name="FashionMNIST",
    dataset_project="fashion-mnist-classification",
    dataset_version="1.0.0",
)

# Add files to the dataset
dataset_path = Path("./dataset")
files = sorted(list(dataset_path.rglob("*")))
print(f"Adding {len(files)} files to clearml dataset")

# # Will add as external files, and will not upload the actual files
# dataset.add_external_files(str(dataset_path), verbose=True)

# Will add as zipped, 512MB chunks
dataset.add_files(str(dataset_path), verbose=True)
dataset.upload()
dataset.finalize()
