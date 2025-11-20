# docx_to_sft_lora.py
import os, json, re, time
from pathlib import Path
import docx
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------- CONFIG ----------
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
}

# ---------- PROMPT TEMPLATE ----------
PROMPT_TEMPLATE = """<|start_of_turn|>user
{instruction}
{input}
<|end_of_turn|>
<|start_of_turn|>assistant
{response}<|end_of_turn|>
"""

# ---------- PII MASK ----------
PII_REPLACERS = [
    (re.compile(r'\b\d{12,19}\b'), '[CARD]'),
    (re.compile(r'\+?\d{7,15}'), '[PHONE]'),
    (re.compile(r'\S+@\S+\.\S+'), '[EMAIL]'),
]

def mask_pii(text: str) -> str:
    if not text: return ""
    t = text
    for pat, repl in PII_REPLACERS:
        t = pat.sub(repl, t)
    return t.strip()

# ---------- DOCX → TEXT ----------
def read_docx(path):
    doc = docx.Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return paragraphs

# ---------- NORMALIZE ENTRY ----------
def make_entry(text):
    instr = mask_pii(text)
    return {
        "instruction": instr,
        "input": "",
        "response": "",
        "template": PROMPT_TEMPLATE.format(instruction=instr, input="", response="")
    }

# ---------- SAVE JSONL ----------
def save_jsonl(entries, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Saved {len(entries)} entries → {out_path}")

# ---------- LOAD JSONL DATASET ----------
def load_jsonl_as_dataset(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            rows.append(json.loads(ln))
    return Dataset.from_list(rows)

# ---------- TOKENIZE & MASK ----------
def tokenize_and_mask(examples, tokenizer, max_length):
    prompts, full_texts = [], []
    for instr, inp, resp, tmpl in zip(examples["instruction"], examples["input"], examples["response"], examples["template"]):
        full = tmpl
        if resp and resp not in full:
            full += resp
        prompt = tmpl.split("<|start_of_turn|>assistant")[0] + "<|start_of_turn|>assistant\n"
        prompts.append(prompt)
        full_texts.append(full)
    tok_full = tokenizer(full_texts, truncation=True, padding="max_length", max_length=max_length)
    tok_prompt = tokenizer(prompts, truncation=True, padding="max_length", max_length=max_length)
    labels = []
    pad_id = tokenizer.pad_token_id
    for i in range(len(full_texts)):
        input_ids = tok_full["input_ids"][i]
        prompt_ids = tok_prompt["input_ids"][i]
        prompt_len = sum(1 for id in prompt_ids if id != pad_id)
        label = [-100]*prompt_len + input_ids[prompt_len:]
        label = label[:max_length] + [-100]*(max_length - len(label))
        labels.append(label)
    tok_full["labels"] = labels
    return tok_full

# ---------- TRAIN LoRA ----------
def train_lora(jsonl_path, cfg=CFG):
    tokenizer = AutoTokenizer.from_pretrained(cfg["MODEL_NAME"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_jsonl_as_dataset(jsonl_path)
    tokenized = ds.map(lambda batch: tokenize_and_mask(batch, tokenizer, cfg["MAX_LENGTH"]),
                       batched=True,
                       remove_columns=["instruction","input","response","template"])

    model = AutoModelForCausalLM.from_pretrained(cfg["MODEL_NAME"], device_map="auto", load_in_4bit=cfg["LOAD_IN_4BIT"])
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
        output_dir=cfg["OUTPUT_DIR"],
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
    trainer.save_model(cfg["OUTPUT_DIR"])
    print("Saved LoRA adapter to:", cfg["OUTPUT_DIR"])

# ---------- MAIN PIPELINE ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", required=True, help="Input DOCX file")
    parser.add_argument("--out_jsonl", required=True, help="Output JSONL file")
    parser.add_argument("--model_name", default=CFG["MODEL_NAME"])
    parser.add_argument("--epochs", type=int, default=CFG["EPOCHS"])
    parser.add_argument("--bs", type=int, default=CFG["BATCH_SIZE"])
    args = parser.parse_args()

    CFG["MODEL_NAME"] = args.model_name
    CFG["EPOCHS"] = args.epochs
    CFG["BATCH_SIZE"] = args.bs

    # 1) Read DOCX
    paras = read_docx(args.docx)
    entries = [make_entry(p) for p in paras]

    # 2) Save JSONL
    save_jsonl(entries, args.out_jsonl)

    # 3) Train LoRA
    train_lora(args.out_jsonl, CFG)

#python docx_to_sft_lora.py --docx thesis.docx --out_jsonl train_template.jsonl --model_name Qwen/Qwen3-4B-Instruct-2507 --epochs 3 --bs 1
