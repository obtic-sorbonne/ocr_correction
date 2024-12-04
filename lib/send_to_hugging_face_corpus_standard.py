#####
# This is the code to create a hugging face dataset and add metrics to the files.
# 
# The ICDAR 2017 dataset was processed with split_corpus.py script. 
# 
# The code also restores the analigned Ground Truth lines. 
####

import os
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from metrics import levenshtein_distance, calculate_wer
from tqdm.auto import tqdm


def retrieve_characters(ocr_aligned, gt_line_aligned):
    '''
    Fonction pour récupérer le texte de ground_truth_aligned en remplaçant les '#' par le texte d'ocr_line.
    ocr_alinegned:        "Bonjouk @e 487Hf monde"
    gt_line_aligned:      "Bonjour le ##### monde"
    ==> GT : "Bonjour le 487Hf monde"
    '''

    result_line = ''.join(
        (ocr_aligned[i] if char == '#' else char) for i, char in enumerate(gt_line_aligned) if i < len(ocr_aligned)
    )
    return result_line

# Load metadata
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
metadata = pd.read_csv("./ICDAR2017-filtered-1800-1900/full_metadata.csv", sep=';')
metadata = metadata.drop(columns=['Type', 'Corpus', 'Split'])

def load_texts_from_folder(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                if len(lines) < 3:
                    print(f"Le fichier {filename} n'a pas assez de lignes.")
                    continue
                
                entry = {
                    "File": filename,
                    "Date": None,  # Placeholder for Date
                    "OCR_toInput": lines[0].replace("[OCR_toInput]", "").strip(),
                    "OCR_aligned": lines[1].replace("[OCR_aligned]", "").strip(),
                    "GS_aligned": lines[2].replace("[ GS_aligned]", "").strip()
                }

                metadata_row = metadata[metadata['File'] == filename].to_dict(orient="records")
                if metadata_row:
                    entry["Date"] = metadata_row[0].get('Date')  # Getting dates from metadata
                    entry.update(metadata_row[0])  
                else:
                    print(f"No metadata found for file: {filename}")

                data.append(entry)
    return data


def calculate_metrics(data):
    for entry in tqdm(data, desc="Calculating metrics"):

        leng_1 = len(entry["OCR_aligned"])
        leng_2 = len(entry["GS_aligned"])

        print(f"OCR_aligned length : {leng_1}, GS_aligned length : {leng_2}") # to make sure there's no error

        entry["Ground_truth_aligned"] = retrieve_characters(entry["OCR_aligned"], entry["GS_aligned"])
        entry["Ground_truth"] = retrieve_characters(entry["OCR_aligned"], entry["GS_aligned"]).replace('@', '') 


        result = levenshtein_distance([entry["Ground_truth"]], [entry["OCR_toInput"]])
        wer = calculate_wer([entry["Ground_truth"]], [entry["OCR_toInput"]])

        entry["distance"] = result["distance"][0]
        entry["cer"] = result["cer"][0]
        entry["wer"] = wer["wer"][0]


train_data = load_texts_from_folder('./ICDAR2017-filtered-1800-1900/train')
dev_data = load_texts_from_folder('./ICDAR2017-filtered-1800-1900/dev')
test_data = load_texts_from_folder('./ICDAR2017-filtered-1800-1900/test')

calculate_metrics(train_data)
calculate_metrics(dev_data)
calculate_metrics(test_data)

train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
dev_dataset = Dataset.from_pandas(pd.DataFrame(dev_data))
test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

dataset_dict = DatasetDict({
    'train': train_dataset,
    'dev': dev_dataset,
    'test': test_dataset
})

api = HfApi()

repo_id = "m-biriuchinskii/ICDAR2017-filtered-1800-1900"
try:
    api.create_repo(repo_id=repo_id, repo_type="dataset")
    print(f"Dépôt {repo_id} créé avec succès.")
except Exception as e:
    print(f"Erreur lors de la création du dépôt : {e}")

try:
    dataset_dict.push_to_hub(repo_id=repo_id, token=huggingface_token)
    print("Dataset uploadé avec succès.")
except Exception as e:
    print(f"Erreur lors de l'upload : {e}")
