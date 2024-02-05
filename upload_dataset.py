import clearml
import pandas as pd
import numpy as np
from pathlib import Path

dataset = clearml.Dataset.create(
    dataset_name="My Dataset",
    dataset_project="My Project",
    dataset_version="1.0.4",
)

# Supported sources:
# - A shared folder: ``/mnt/share/folder``
# - S3: ``s3://bucket/folder``
# - Google Cloud Storage: ``gs://bucket-name/folder``
# - Azure Storage: ``azure://company.blob.core.windows.net/folder/``
dataset_path = Path("./dataset")

files = sorted(list(dataset_path.rglob("*")))

print(f"Adding {len(files)} files to clearml dataset")

# Will add as external files, and will not upload the actual files
# dataset.add_external_files(str(dataset_path), verbose=True)

# Will add as zipped, 512MB chunks
dataset.add_files(str(dataset_path), verbose=True)

dataset.upload()
dataset.finalize()

# Add metadata or artifacts
dataset_train_task: clearml.Task = clearml.Task.get_task(dataset.id)
dataset_train_task.set_comment("This is a comment, it will be displayed in the UI")

# Create a random pandas dataframe and upload it as an artifact


df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))
dataset_train_task.get_logger().report_table(
    "Random DataFrame", series="DataFrames", iteration=0, table_plot=df
)