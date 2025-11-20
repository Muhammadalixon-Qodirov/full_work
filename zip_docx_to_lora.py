# zip_docx_to_lora.py
import os
import zipfile
import json
import re
import time
from pathlib import Path
import argparse

import docx
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -----------------------------
# CONFIG (o'zgartiring)
# -----------------------------
CFG = {
    "MODEL_NAME": "Qwen/Qwen3-4B-Instruct-2507",
    "LOAD_IN_4BIT": True,
    "MAX_LENGTH": 2048,
    "BATCH_SIZE": 1,
    "EPOCHS": 3,
    "LR": 2e-4,
    "LORA_R": 16,
    "LORA_ALPHA": 32,
    "LORA_DROPOUT": 0.05,
    "LORA_TARGET_MODULES": ["q_proj","v_proj","k_proj","o_proj"],
    "OUTPUT_DIR": "lora_out",
    "TEMP_DIR": "tmp_docs"
}

# -----------------------------
# 1) PII masking + clean text
# -----------------------------
PII_PATTERNS = [
    (re.compile(r'\b\d{12,19}\b'), '[CARD]'),
    (re.compile(r'\+?\d{7,15}'), '[PHONE]'),
    (re.compile(r'\S+@\S+\.\S+'), '[EMAIL]'),
]

def mask_pii(text: str) -> str:
    t = text
    for pat, repl in PII_PATTERNS:
        t = pat.sub(repl, t)
    return t.strip()

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', mask_pii(text)).strip()

# -----------------------------
# 2) DOCX reader
# -----------------------------
def read_docx(path):
    doc = docx.Document(path)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return [clean_text(p) for p in paras]

# -----------------------------
# 3) Instruction-Response template
# -----------------------------
PROMPT_TEMPLATE = """<|start_of_turn|>user
{instruction}
{input}
<|end_of_turn|>
<|start_of_turn|>assistant
{response}<|end_of_turn|>
"""

def make_entry(paragraph):
    return {
        "instruction": paragraph,
        "input": "",
        "response": paragraph,  # aynan paragrafni javob sifatida ham qo‘yamiz
        "template": PROMPT_TEMPLATE.format(instruction=paragraph, input="", response=paragraph)
    }

# -----------------------------
# 4) Convert ZIP of DOCX to JSONL
# -----------------------------
def convert_zip_to_jsonl(zip_path, out_jsonl, temp_dir=CFG["TEMP_DIR"]):
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(temp_dir)
    
    all_entries = []
    for docx_file in temp_dir.rglob("*.docx"):
        paras = read_docx(docx_file)
        for p in paras:
            all_entries.append(make_entry(p))
    
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    
    print(f"Converted {len(all_entries)} paragraphs -> {out_jsonl}")
    return out_jsonl

# -----------------------------
# 5) Tokenize + mask prompt (for SFT/LoRA)
# -----------------------------
def tokenize_and_mask(examples, tokenizer, max_length):
    prompts = []
    full_texts = []
    for tmpl in examples["template"]:
        full_texts.append(tmpl)
        prompt = tmpl.split("<|start_of_turn|>assistant")[0] + "<|start_of_turn|>assistant\n"
        prompts.append(prompt)
    tok_full = tokenizer(full_texts, truncation=True, padding="max_length", max_length=max_length)
    tok_prompt = tokenizer(prompts, truncation=True, padding="max_length", max_length=max_length)
    
    labels = []
    pad_id = tokenizer.pad_token_id
    for i in range(len(full_texts)):
        input_ids = tok_full["input_ids"][i]
        prompt_ids = tok_prompt["input_ids"][i]
        prompt_len = sum(1 for id in prompt_ids if id != pad_id)
        label = [-100]*prompt_len + input_ids[prompt_len:]
        label = label[:max_length] + [-100]*max(0, max_length - len(label))
        labels.append(label)
    tok_full["labels"] = labels
    return tok_full

# -----------------------------
# 6) Load JSONL
# -----------------------------
def load_jsonl_dataset(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            rows.append(json.loads(ln))
    return Dataset.from_list(rows)

# -----------------------------
# 7) Train LoRA
# -----------------------------
def train_lora(jsonl_path, out_dir, cfg=CFG):
    tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_NAME"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_jsonl_dataset(jsonl_path)
    tokenized = ds.map(lambda batch: tokenize_and_mask(batch, tokenizer, cfg["MAX_LENGTH"]),
                       batched=True, remove_columns=["instruction","input","response","template"])
    
    model = AutoModelForCausalLM.from_pretrained(cfg["MODEL_NAME"],
                                                 device_map="auto",
                                                 load_in_4bit=cfg["LOAD_IN_4BIT"])
    model = prepare_model_for_kbit_training(model)
    
    lora_conf = LoraConfig(
        r=cfg["LORA_R"],
        lora_alpha=cfg["LORA_ALPHA"],
        target_modules=cfg["LORA_TARGET_MODULES"],
        lora_dropout=cfg["LORA_DROPOUT"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_conf)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=cfg["BATCH_SIZE"],
        gradient_accumulation_steps=1,
        num_train_epochs=cfg["EPOCHS"],
        learning_rate=cfg["LR"],
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    trainer.train()
    trainer.save_model(out_dir)
    print("Saved LoRA adapter to:", out_dir)

# -----------------------------
# 8) Orchestrator
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Input zip containing DOCX files")
    parser.add_argument("--out_jsonl", default="train_all.jsonl", help="Output JSONL path")
    parser.add_argument("--lora_out", default=f"lora_{int(time.time())}", help="LoRA output directory")
    parser.add_argument("--model", default=CFG["MODEL_NAME"])
    parser.add_argument("--epochs", type=int, default=CFG["EPOCHS"])
    parser.add_argument("--bs", type=int, default=CFG["BATCH_SIZE"])
    parser.add_argument("--4bit", action="store_true", help="Load model in 4bit")
    args = parser.parse_args()

    CFG["MODEL_NAME"] = args.model
    CFG["EPOCHS"] = args.epochs
    CFG["BATCH_SIZE"] = args.bs
    CFG["LOAD_IN_4BIT"] = args.4bit

    jsonl_file = convert_zip_to_jsonl(args.zip, args.out_jsonl)
    train_lora(jsonl_file, args.lora_out, CFG)


#python zip_docx_to_lora.py --zip docs.zip --lora_out my_lora_adapter --4bit
