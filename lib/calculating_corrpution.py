import json
from datasets import load_dataset
from scrambledtext import ProbabilityDistributions

dataset = load_dataset("m-biriuchinskii/ICDAR2017-filtered-1800-1900-5")

aligned_texts = [
    (row["Sentence_GT_aligned"], row["Sentence_OCR_aligned"])
    for row in dataset['train'] 
]

probs = ProbabilityDistributions(aligned_texts)
probs.save_to_json('probabilities-ICDAR2017-filtered-1800-1900-5.json')
