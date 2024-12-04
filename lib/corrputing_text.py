from scrambledtext import ProbabilityDistributions, CorruptionEngine
from datasets import load_dataset
from huggingface_hub import HfApi
import os

# Load pre-calculated probability distributions
probs = ProbabilityDistributions.load_from_json('probabilities-ICDAR2017-filtered-1800-1900-5.json')

# Create a corruption engine
engine = CorruptionEngine(
    probs.conditional,
    probs.substitutions,
    probs.insertions,
    target_wer=0.2,
    target_cer=0.1
)

# Upload the dataset
api = HfApi()
repo_id = "m-biriuchinskii/ICDAR2017-filtered-1800-1900-5"  
new_repo_id = "m-biriuchinskii/ICDAR2017-filtered-1800-1900-6"  # New repository name

dataset = load_dataset(repo_id)

def add_corrupted_data(example):
    try:
        corrupted_text, corrupted_wer, corrupted_cer, _ = engine.corrupt_text(example['Sentence_OCR'])
        example['Sentence_OCR_corrupted'] = corrupted_text.replace('@', '')
        example['corrupted_cer'] = corrupted_cer
        example['corrupted_wer'] = corrupted_wer
    except ZeroDivisionError:
        example['Sentence_OCR_corrupted'] = example['Sentence_OCR']
        example['corrupted_cer'] = 0
        example['corrupted_wer'] = 0
    return example

# Apply the corruption process to all splits
for split in dataset.keys():
    dataset[split] = dataset[split].map(add_corrupted_data, num_proc=4)


try:
    api.create_repo(repo_id=new_repo_id, repo_type="dataset", private=False)
    print(f"New repository '{new_repo_id}' created.")
    
    dataset.push_to_hub(repo_id=new_repo_id, token=os.getenv('HUGGINGFACE_TOKEN'))
    print("Dataset successfully uploaded with new columns to the new repository.")
except Exception as e:
    print(f"Error uploading dataset: {e}")
