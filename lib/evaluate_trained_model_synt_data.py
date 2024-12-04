import json
import os
import re

import torch
from datasets import load_dataset
from huggingface_hub import login
from metrics import (
    calculate_cer_reduction,
    calculate_pcis,
    calculate_wer,
    levenshtein_distance,
)
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

token = os.getenv("HUGGINGFACE_TOKEN")
if token is None:
    print("Error: HUGGINGFACE_TOKEN environment variable not set.")
else:
    login(token=token)

# Load the dataset, model, and the tokenizer

dataset = load_dataset("m-biriuchinskii/ICDAR2017-filtered-1800-1900-6")
model = AutoModelForCausalLM.from_pretrained("m-biriuchinskii/Llama-3.2-3B-ocr-correction-3-instruction-corrected-mixed-data")
tokenizer = AutoTokenizer.from_pretrained("m-biriuchinskii/Llama-3.2-3B-ocr-correction-3-instruction-corrected-mixed-data")

# Check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prompt model 
def call_the_model(ocr_text, max_attempts=3):
    """
    Call the model to generate a corrected OCR text, with retry and validation logic.
    
    Parameters:
        ocr_text (str): The OCR text to be corrected.
        max_attempts (int): Maximum number of attempts to generate valid output.
    
    Returns:
        str: The extracted corrected text enclosed in <RESULT> tags, or an error message.
    """
    instruction = """Tu effectues une tâche de correction du texte océrisé. 
1. Corriger les erreurs OCR (lettres manquantes, accents) du texte de la bibliothèque numérique Gallica fourni.
2. Maintenir le style et le langage du XIXe siècle, en évitant la modernisation.
3. Ne pas altérer les marques de coupe de ligne (par exemple, seu-lement, oc-tobre).
4. Retourner le texte corrigé uniquement entre les balises "<RESULT></RESULT>".
"""    

    messages = [{"role": "system", "content": instruction},
                {"role": "user", "content": ocr_text}]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    for attempt in range(max_attempts):
        try:
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}  # Move tensors to the correct device
            
            # Generate output from the model
            outputs = model.generate(**inputs, max_new_tokens=1220, num_return_sequences=1)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Validate and extract the output
            pattern = r'<RESULT>(.*?)<\/RESULT>'
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[-1].strip()  # Return the last valid result
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        
        print(f"Attempt {attempt + 1} failed: Retrying...")

    return f"Failed after {max_attempts} attempts. OUTPUT: {text.split("assistant")[1]}"


def generate_json_with_predictions(dataset, split, output_path="output.json", min_length=15, max_length=1300):
    """
    Generate JSON with OCR text, ground truth, model predictions, CER, and WER for a specified dataset split.
    """
    data = []
    num_examples = len(dataset[split]["Sentence_OCR_corrupted"])

    for i in tqdm(range(num_examples), desc=f"Processing {split} examples"):
        ocr_text = dataset[split]["Sentence_OCR_corrupted"][i]
        gs_text = dataset[split]["Sentence_GT"][i]

        if min_length <= len(ocr_text) <= max_length and min_length <= len(gs_text) <= max_length:
            model_prediction = call_the_model(ocr_text)

            cer_result = levenshtein_distance([gs_text], [model_prediction])["cer"].iloc[0]

            wer_result = calculate_wer([gs_text], [model_prediction])["wer"].iloc[0]

            dataset_cer = dataset[split]["corrupted_cer"][i]
            dataset_wer = dataset[split]["corrupted_wer"][i]

            cer_reduction = calculate_cer_reduction(dataset_cer, cer_result)

            pcis = calculate_pcis(cer_result, dataset_cer)

            data.append({
                "File": dataset[split]["File"][i],
                "Data": {
                    "OCR_toInput": ocr_text,
                    "Ground_truth": gs_text,
                    "Model_prediction": model_prediction 
                },
                
                "Metrics": {
                    "Dataset сer": dataset_cer,
                    "Model cer": cer_result,
                    "Dataset wer": dataset_wer,
                    "Model wer": wer_result,
                    "cer reduction %": cer_reduction,
                    "pcis":pcis
                }
            })

    if output_path:
        with open(output_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    return data

generate_json_with_predictions(dataset, 'test', output_path="Llama-3.2-3B-ocr-correction-3-instruction-corrected-mixed-data-synth-evaluation.json")
