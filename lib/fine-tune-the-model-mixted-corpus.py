import os
import random

import torch
import wandb
import bitsandbytes as bnb
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig

token = os.getenv("HUGGINGFACE_TOKEN")
if token is None:
    raise ValueError("Error: HUGGINGFACE_TOKEN environment variable not set.")
else:
    login(token=token)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

wandb.login(key=os.getenv("WANDB_API_KEY"))

run = wandb.init(
    project="Fine-tune_LLaMA3.2_on_OCR_Correction", 
    job_type="training", 
    anonymous="allow"
)

base_model = "meta-llama/Llama-3.2-3B-Instruct"
new_model = "Llama-3.2-3B-ocr-correction-3-instruction-corrected-mixed-data-full-parameters"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

torch.cuda.empty_cache()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print(torch.cuda.memory_summary())

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)
print(torch.cuda.memory_summary())

# Load tokenizer, dataset
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
dataset_name = "m-biriuchinskii/ICDAR2017-filtered-1800-1900-6"
dataset = load_dataset(dataset_name)


# Split dataset into two subsets
def split_dataset(dataset):
    # Split into two parts (80% Sentence_OCR, 20% Sentence_OCR_corrupted)
    dataset_ocr = dataset["train"].filter(lambda row, idx: random.random() < 0.8, with_indices=True)
    dataset_ocr_corrupted = dataset["train"].filter(lambda row, idx: random.random() >= 0.8, with_indices=True)
    
    return dataset_ocr, dataset_ocr_corrupted

dataset_ocr, dataset_ocr_corrupted = split_dataset(dataset)

def format_chat_template(row):
    if row in dataset_ocr:
        content = row["Sentence_OCR"]
    else:
        content = row["Sentence_OCR_corrupted"]

    instruction = """Tu effectues une tâche de correction du texte océrisé. 
1. Corriger les erreurs OCR (lettres manquantes, accents) du texte de la bibliothèque numérique Gallica fourni.
2. Maintenir le style et le langage du XIXe siècle, en évitant la modernisation.
3. Ne pas altérer les marques de coupe de ligne (par exemple, seu-lement, oc-tobre).
4. Retourner le texte corrigé uniquement entre les balises "<RESULT></RESULT>".
"""
    response = f"<RESULT>{row['Sentence_GT']}</RESULT>"
    prompt = f"""{content}"""
    row_json = [{"role": "system", "content": instruction },
               {"role": "user", "content": prompt},
               {"role": "assistant", "content": response}]
    
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

# Map the function over the dataset
torch.cuda.empty_cache()

dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)

dataset_train = dataset["train"].shuffle(seed=65) #select(range(650)) ### For test
dataset_eval = dataset["dev"].shuffle(seed=65) #.select(range(336)) 
dataset_test = dataset["test"].shuffle(seed=65) #.select(range(301)) 

for i in range(3): 
    print(dataset_train[i]["text"])


print(torch.cuda.memory_summary())
torch.cuda.empty_cache()

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit precision
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

modules = find_all_linear_names(model)


# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules
)
model = get_peft_model(model, peft_config)

# Check if CUDA is available 
use_bf16 = torch.cuda.get_device_capability()[0] >= 8  
use_fp16 = torch.cuda.is_available() and not use_bf16  

tokenizer.pad_token = tokenizer.eos_token

training_arguments = SFTConfig(
    output_dir=new_model,
    run_name="fine_tune_ocr_correction",
    per_device_train_batch_size=4, # max 4 batches
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16, # the bigger the better for GPUs
    optim="paged_adamw_32bit",
    num_train_epochs=8, 
    eval_strategy="steps",
    eval_steps=30,  
    save_steps=30,
    logging_steps=10,  
    warmup_steps=100,
    logging_strategy="steps",
    learning_rate= 5e-5, # 5e-5 = 0.00005 ; 2e-4 = 0.0002, 
    fp16=use_fp16, 
    bf16=use_bf16,  
    group_by_length=True,
    report_to="wandb",
    max_seq_length=1220,
    save_strategy="steps",
    dataset_text_field="text",
    max_grad_norm=1.0,
    warmup_ratio=0.05,
    load_best_model_at_end = True
)

# EarlyStoppingCallback setup
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=4,  # Number of evaluation steps with no improvement to wait
    early_stopping_threshold=0.0, # Minimum improvement to trigger early stopping (e.g., 0.0 means any improvement)
)

torch.cuda.empty_cache()

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    callbacks=[early_stopping_callback],  
)

# Start training
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())

trainer.train()
torch.cuda.empty_cache()

wandb.finish()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
trainer.model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model)
