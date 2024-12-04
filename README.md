# Training Llama 3.2-Instruct for Historical French Post-OCR Correction using Synthetic Data

This repository contains the code and resources for the paper **"Training Llama 3.2-Instruct for Historical French Post-OCR Correction using Synthetic Data: A Case Study"** presented by Mikhail Biriuchinskii, Motasem Alrahabi, and Glenn Roe from the ObTIC Lab, Sorbonne University.

## Abstract
With the rise of generative AI, this project explores using LLMs to correct OCR errors in 19th-century French texts. The study evaluates the Llama-3.2-Instruct model using real and synthetic datasets, highlighting the challenges of generalization and limitations in improving OCR corrections.

For full details, refer to the [article](LINK NOT READY YET).

## Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/MichaBiriuchinskii/ObTIC-ocr-correction.git
cd ObTIC-ocr-correction
```
   
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies, and [scrambledtex library](https://github.com/JonnoB/scrambledtext/tree/main):
```bash
pip install -r requirements.txt
python -m pip install git+https://github.com/JonnoB/scrambledtext
```
## Data

The final dataset used for fine-tuning in this project is deployed on Hugging Face Datasets: [ICDAR2017-filtered-1800-1900-6](https://huggingface.co/datasets/m-biriuchinskii/ICDAR2017-filtered-1800-1900-6).

The Data Processing Pipeline:
```txt
ICDAR2017 (original dataset)
     │
     ▼
Filter on Dates, Doc Type, and Language
     │
     ▼
ICDAR2017-filtered-1800-1900 
     │
     ▼
GT Phrases Retrieval, Tokenization into Phrases with 85% Similarity Threshold
     │
     ▼
ICDAR2017-filtered-1800-1900-5
     │
     ▼
Only 15% of Dataset with Zero CER Cases
     │
     ▼
Add Corruption Data
     │
     ▼
ICDAR2017-filtered-1800-1900-6

```

## Library Code Description

- **lib/split_corpus.py**: Divides the filtered dataset (data/ICDAR2017-filtered-1800-1900.zip) into three subsets: test, development, and training data.
  
- **lib/send_to_hugging_face_corpus_standard.py**: Creates the first version of the dataset without tokenization of the phrases. This version is available on Hugging Face: [ICDAR2017-filtered-1800-1900](https://huggingface.co/datasets/m-biriuchinskii/ICDAR2017-filtered-1800-1900).

- **lib/send_to_hugging_face_corpus_tokenized_filtered.py**: Tokenizes the first version of the dataset into phrases and applies an 85% similarity threshold.

- **clean_tokenized_corpus_15_percent.py**: Filters out cases where the Character Error Rate (CER) is 0, ensuring that only 15% of the dataset contains cases with zero CER.

- **calculating_corrpution.py**: Script to calculate corruption percentages between OCR and ground truth (GT) phrases in the dataset.

- **corrputing_text.py**: Based on the corruption percentages calculated earlier, this script introduces realistic errors to create a "Sentence_OCR_corrupted" column in the dataset, giving us the final version of the dataset - [ICDAR2017-filtered-1800-1900-6](https://huggingface.co/datasets/m-biriuchinskii/ICDAR2017-filtered-1800-1900-6).

- **lib/metrics.py**: Contains functions to calculate various performance metrics.

- **lib/fine-tune-the-model.py**: Main script for fine-tuning the model on either real or synthetic data.

- **lib/fine-tune-the-model-mixted-corpus.py**: Main script for fine-tuning the model using a mixed dataset with an 80/20 split between real and synthetic data.

- **evaluate_trained_model_synt_data.py**: Script to evaluate the trained model using synthetic data.

- **evaluate_trained_model_on_real_data.py**: Script to evaluate the trained model using real data.

- **calculate_metrics_for_json.py**: Script that calculates median metrics based on a model's output in JSON format.
- **lib/visualisation.ipynb**: Notebook to create visualisations

## Models 
- [Llama-3.2-3B-post-ocr-correction-real-data](https://huggingface.co/m-biriuchinskii/Llama-3.2-3B-ocr-correction-3-instruction-corrected-real-data-full-params)
- [Llama-3.2-3B-post-ocr-correction-synthetic-data](https://huggingface.co/m-biriuchinskii/Llama-3.2-post-ocr-synthetic-data-2)
- [Llama-3.2-3B-post-ocr-correction-mixed-data](https://huggingface.co/m-biriuchinskii/Llama-3.2-3B-ocr-correction-3-instruction-corrected-mixed-data)

## Results 

```txt
                  Base Model       Synth Model       Mixed Model
Evaluation      Real  Synthetic   Real  Synthetic   Real  Synthetic
----------------------------------------------------------------------
Avg Dataset CER 0.0139 0.0984     0.0139 0.0984     0.0139 0.0984
Avg Model CER   0.0175 0.1922     0.0220 0.1526     0.0198 0.1921
Avg Dataset WER 0.0621 0.2199     0.0621 0.2199     0.0621 0.2199
Avg Model WER   0.0819 1.0188     0.1026 0.8028     0.0872 1.0169
Avg PCIS        -0.0038 -0.0973   -0.0084 -0.0604   -0.0062 -0.1047

```
## Citation
If you use this code or find it helpful, please cite the article:

```arduino
Mikhail Biriuchinskii, Motasem Alrahabi, Glenn Roe. "Training Llama 3.2-Instruct for Historical French Post-OCR Correction using Synthetic Data." Sorbonne University, 2024.
```

## Licence
This project is licensed under the MIT License - see the LICENSE file for details.

