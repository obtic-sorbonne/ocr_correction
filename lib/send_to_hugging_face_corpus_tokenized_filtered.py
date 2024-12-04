import os
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from metrics import levenshtein_distance, calculate_wer
from tqdm.auto import tqdm
from fuzzywuzzy import fuzz
import nltk
import pysbd
from concurrent.futures import ThreadPoolExecutor

# Configurable similarity threshold
SIMILARITY_THRESHOLD = 85

# Initialize segmenter or download nltk data
segmenter = pysbd.Segmenter(language="en", clean=True)

def load_metadata(metadata_path):
    metadata = pd.read_csv(metadata_path, sep=';')
    metadata = metadata.drop(columns=['Type', 'Corpus', 'Split'])
    return metadata.set_index("File").to_dict(orient="index")

metadata = load_metadata("./ICDAR2017-filtered-1800-1900/full_metadata.csv")

def load_texts_from_folder(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                if len(lines) < 3:
                    continue

                entry = {
                    "File": filename,
                    "Region_OCR": lines[0].replace("[OCR_toInput]", "").strip(),
                    "OCR_aligned": lines[1].replace("[OCR_aligned]", "").strip(),
                    "GS_aligned": lines[2].replace("[ GS_aligned]", "").strip(),
                }
                entry.update(metadata.get(filename, {"Date": None}))
                data.append(entry)
    return data

def retrieve_characters(ocr_aligned, gt_line_aligned):
    '''
    Fonction pour récupérer le texte de ground_truth_aligned en remplaçant les '#' par le texte d'ocr_line.
    ocr_alinegned:        "Bonjouk le 487Hf monde"
    gt_line_aligned:      "Bonjour le ##### monde"
    ==> GT : "Bonjour le 487Hf monde"
    '''
    return ''.join(
        (ocr_aligned[i] if char == '#' else char) 
        for i, char in enumerate(gt_line_aligned) if i < len(ocr_aligned)
    )

# Attempt to align sentences multiple times before skipping
def attempt_sentence_alignment(entry, max_attempts=10):
    for attempt in range(max_attempts):
        # Tokenize sentences with PySBD
        ocr_sentences = segmenter.segment(entry["OCR_aligned"] .replace('@', ''))
        gt_sentences = segmenter.segment(entry["Ground_truth_aligned"] .replace('@', ''))

        # If counts match, return the tokenized sentences
        if len(ocr_sentences) == len(gt_sentences):
            return ocr_sentences, gt_sentences

        # Log retry attempt
        print(f"Retrying segmentation for {entry['File']} (attempt {attempt + 1})")
    
    # If all attempts fail, return None to indicate a mismatch
    print(f"Skipping {entry['File']} after {max_attempts} attempts due to mismatched sentence counts.")
    return None, None


def align_texts_with_at(ocr_text, gt_text):
    """
    Aligns two texts at the character level by inserting `@` symbols where necessary 
    to make their lengths match while preserving alignment.

    Args:
        ocr_text (str): The OCR text.
        gt_text (str): The ground truth text.

    Returns:
        tuple: The aligned OCR text and aligned ground truth text with `@` inserted.
    """
    aligned_ocr = []
    aligned_gt = []
    ocr_len, gt_len = len(ocr_text), len(gt_text)
    
    i, j = 0, 0  # Pointers for OCR and GT text

    while i < ocr_len or j < gt_len:
        ocr_char = ocr_text[i] if i < ocr_len else None
        gt_char = gt_text[j] if j < gt_len else None

        if ocr_char == gt_char:  # Characters match
            aligned_ocr.append(ocr_char)
            aligned_gt.append(gt_char)
            i += 1
            j += 1
        elif ocr_char is None:  # OCR text is shorter
            aligned_ocr.append('@')
            aligned_gt.append(gt_char)
            j += 1
        elif gt_char is None:  # GT text is shorter
            aligned_ocr.append(ocr_char)
            aligned_gt.append('@')
            i += 1
        else:  # Characters differ, add `@` to align
            aligned_ocr.append(ocr_char)
            aligned_gt.append(gt_char)
            i += 1
            j += 1

    return ''.join(aligned_ocr), ''.join(aligned_gt)

# Process a single entry by calculating metrics for matched sentences
def process_entry(entry):
    entry["Ground_truth_aligned"] = retrieve_characters(entry["OCR_aligned"], entry["GS_aligned"])
    entry["Ground_truth"] = entry["Ground_truth_aligned"].replace('@', '')

    # Attempt sentence alignment up to 10 times
    ocr_sentences, gt_sentences = attempt_sentence_alignment(entry)
    if ocr_sentences is None or gt_sentences is None:
        return []  # Skip entry if alignment failed

    processed_data = []
    for ocr_sentence, gt_sentence in zip(ocr_sentences, gt_sentences):
        if len(ocr_sentence) < 15 or len(gt_sentence) < 15:
            continue

        similarity_score = fuzz.token_sort_ratio(ocr_sentence, gt_sentence)
        if similarity_score < SIMILARITY_THRESHOLD:
            continue

        result = levenshtein_distance([gt_sentence], [ocr_sentence])
        wer = calculate_wer([gt_sentence], [ocr_sentence])

        ocr_sentence_aligned, gt_sentence_aligned = align_texts_with_at(ocr_sentence, gt_sentence)

        processed_data.append({
            "File": entry["File"],
            "Date": entry["Date"],
            "Region_OCR": entry["Region_OCR"],
            "Region_OCR_aligned": entry["OCR_aligned"],
            "Region_GT_aligned": entry["Ground_truth_aligned"],
            "Sentence_OCR_aligned": ocr_sentence_aligned,
            "Sentence_GT_aligned": gt_sentence_aligned,
            "Sentence_OCR": ocr_sentence,
            "Sentence_GT": gt_sentence,
            "Distance": result["distance"][0],
            "CER": result["cer"][0],
            "WER": wer["wer"][0]
        })
    return processed_data

# Batch process data entries using multiprocessing
def calculate_metrics(data):
    processed_data = []
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_entry, data), total=len(data), desc="Calculating metrics"))
        for result in results:
            processed_data.extend(result)
    return processed_data
# Load and process each data split
train_data = calculate_metrics(load_texts_from_folder('./ICDAR2017-filtered-1800-1900/train'))
dev_data = calculate_metrics(load_texts_from_folder('./ICDAR2017-filtered-1800-1900/dev'))
test_data = calculate_metrics(load_texts_from_folder('./ICDAR2017-filtered-1800-1900/test'))

# Convert processed data to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
dev_dataset = Dataset.from_pandas(pd.DataFrame(dev_data))
test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

# Create DatasetDict and push to Hugging Face Hub
dataset_dict = DatasetDict({
    'train': train_dataset,
    'dev': dev_dataset,
    'test': test_dataset
})

api = HfApi()
repo_id = "m-biriuchinskii/ICDAR2017-filtered-1800-1900-5"
try:
    api.create_repo(repo_id=repo_id, repo_type="dataset")
except Exception as e:
    print(f"Error creating repo: {e}")

try:
    dataset_dict.push_to_hub(repo_id=repo_id, token=os.getenv('HUGGINGFACE_TOKEN'))
    print("Dataset successfully uploaded.")
except Exception as e:
    print(f"Error uploading dataset: {e}")