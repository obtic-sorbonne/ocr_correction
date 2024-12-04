import json
import math
import numpy as np

file_path = "Llama-3.2-3B-ocr-correction-3-instruction-corrected-real-data-full-params-synth-data-eval.json"

with open(file_path, "r") as file:
    data = json.load(file)

valid_pcis = []
valid_dataset_cers = []
valid_model_cers = []
valid_dataset_wers = []
valid_model_wers = []

for entry in data:
    metrics = entry.get("Metrics", {})
    dataset_cer = metrics.get("Dataset сer", None)
    model_cer = metrics.get("Model cer", None)
    dataset_wer = metrics.get("Dataset wer", None)
    model_wer = metrics.get("Model wer", None)
    pcis = metrics.get("pcis", None)
    
    if pcis is not None and not math.isnan(pcis) and not math.isinf(pcis):
        valid_pcis.append(pcis)
    if dataset_cer is not None and not math.isnan(dataset_cer) and not math.isinf(dataset_cer):
        valid_dataset_cers.append(dataset_cer)
    if model_cer is not None and not math.isnan(model_cer) and not math.isinf(model_cer):
        valid_model_cers.append(model_cer)
    if dataset_wer is not None and not math.isnan(dataset_wer) and not math.isinf(dataset_wer):
        valid_dataset_wers.append(dataset_wer)
    if model_wer is not None and not math.isnan(model_wer) and not math.isinf(model_wer):
        valid_model_wers.append(model_wer)

# Calculate averages for PCIS, CERs, and WERs
average_pcis = np.mean(valid_pcis) if valid_pcis else None
average_dataset_cer = np.mean(valid_dataset_cers) if valid_dataset_cers else None
average_model_cer = np.mean(valid_model_cers) if valid_model_cers else None
average_dataset_wer = np.mean(valid_dataset_wers) if valid_dataset_wers else None
average_model_wer = np.mean(valid_model_wers) if valid_model_wers else None


# Display the averages
print(f"{file_path}")
print(f"Average PCIS: {average_pcis:.8f}" if average_pcis is not None else "Average PCIS: No valid data")
print(f"Average Dataset CER: {average_dataset_cer:.8f}" if average_dataset_cer is not None else "Average Dataset CER: No valid data")
print(f"Average Model CER: {average_model_cer:.8f}" if average_model_cer is not None else "Average Model CER: No valid data")
print(f"Average Dataset WER: {average_dataset_wer:.8f}" if average_dataset_wer is not None else "Average Dataset WER: No valid data")
print(f"Average Model WER: {average_model_wer:.8f}" if average_model_wer is not None else "Average Model WER: No valid data")
