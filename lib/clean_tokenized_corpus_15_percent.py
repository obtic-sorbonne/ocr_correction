import pandas as pd
import os
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi

# Define the Hugging Face dataset repository ID
repo_id = "m-biriuchinskii/ICDAR2017-filtered-1800-1900-5"
MAX_ZERO_DISTANCE_PERCENTAGE = 15

dataset_dict = load_dataset(repo_id)

def clean_split(dataset):
    df = dataset.to_pandas()

    # Separate zero-distance rows and non-zero-distance rows
    zero_distance_rows = df[df["Distance"] == 0]
    non_zero_distance_rows = df[df["Distance"] != 0]

    # Calculate the maximum number of zero-distance rows allowed
    total_count = len(df)
    max_zero_distance_rows = int((MAX_ZERO_DISTANCE_PERCENTAGE / 100) * total_count)

    # If we have more zero-distance rows than allowed, sample to reduce
    if len(zero_distance_rows) > max_zero_distance_rows:
        zero_distance_rows = zero_distance_rows.sample(n=max_zero_distance_rows, random_state=42)

    # Combine filtered zero-distance rows with non-zero-distance rows
    cleaned_df = pd.concat([zero_distance_rows, non_zero_distance_rows]).sample(frac=1, random_state=42).reset_index(drop=True)

    return Dataset.from_pandas(cleaned_df)

# Clean each split in the dataset
cleaned_splits = {split: clean_split(dataset_dict[split]) for split in dataset_dict.keys()}

cleaned_dataset_dict = DatasetDict(cleaned_splits)

api = HfApi()
try:
    cleaned_dataset_dict.push_to_hub(repo_id=repo_id, token=os.getenv('HUGGINGFACE_TOKEN'))
    print(f"Cleaned dataset successfully uploaded to {repo_id}")
except Exception as e:
    print(f"Error uploading cleaned dataset: {e}")
