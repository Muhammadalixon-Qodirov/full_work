
---

## **1️⃣ Skriptning vazifasi umumiy**

Bu skript **“DOCX zip → JSONL → LoRA fine-tuning”** pipeline’ni avtomatlashtiradi.

Ya’ni:

1. Siz **DOCX fayllarni zip** qilasiz (masalan, 50 ta fayl).
2. Skript zip’ni ochadi, har bir DOCX ichidagi matnni o‘qiydi.
3. Matnni **tozalaydi** (bo‘sh joylar, PII: telefon, email, kart raqami).
4. Har paragrafni **instruction-response template** ga aylantiradi.
5. Barcha paragraflar JSONL formatida saqlanadi.
6. Shu JSONL asosida **LoRA adapter** yaratiladi va model fine-tuning qilinadi.




| Qadam                                  | Vazifasi                                                                                               | Natija                                        |             |               |                                                                   |                                                         |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------------- | ----------- | ------------- | ----------------------------------------------------------------- | ------------------------------------------------------- |
| 1️⃣ Zip faylni ochish                  | Siz bergan `.zip` fayldagi barcha DOCX fayllarni `temp` papkaga chiqaradi                              | Barcha DOCX fayllar alohida ochiladi          |             |               |                                                                   |                                                         |
| 2️⃣ DOCX matnini o‘qish                | Har DOCX fayldagi har paragrafni olib, bo‘sh joylarni, PII (telefon, email, karta raqami) ni tozalaydi | Har paragraf **tozalangan matn** bo‘ladi      |             |               |                                                                   |                                                         |
| 3️⃣ Template yaratish                  | Har paragrafni `<                                                                                      | start_of_turn                                 | >user ... < | start_of_turn | >assistant` shaklida instruction-response template ga aylantiradi | JSONL uchun tayyor **instruction + response** struktura |
| 4️⃣ JSONL fayl yaratish                | Barcha paragraflarni `.jsonl` faylga yozadi                                                            | Bir faylda barcha training datalari saqlanadi |             |               |                                                                   |                                                         |
| 5️⃣ Tokenization va labels tayyorlash  | Model uchun prompt va response tokenlarga ajratiladi, prompt tokenlari `-100` qilinadi                 | Model faqat javob qismi bo‘yicha o‘rganadi    |             |               |                                                                   |                                                         |
| 6️⃣ Modelni yuklash va LoRA tayyorlash | Asosiy model yuklanadi, LoRA adapter qo‘shiladi                                                        | RAM tejamkor, asosiy model o‘zgarmaydi        |             |               |                                                                   |                                                         |
| 7️⃣ Training                           | JSONL dagi matnlar bo‘yicha LoRA adapter o‘qitiladi                                                    | Tayyor **LoRA adapter** hosil bo‘ladi         |             |               |                                                                   |                                                         |
| 8️⃣ Natija                             | `lora_out/` papkada adapter, JSONL fayl                                                                | Boshqa modelga yuklab inference qilsa bo‘ladi |             |               |                                                                   |                                                         |


Natija: Sizda **tayyor LoRA adapter** bo‘ladi, uni boshqa model bilan inference uchun ishlatish mumkin.

---

## **2️⃣ Bosqichma-bosqich tushuntirish**

### **Step 1 — ZIP faylni ochish**

```python
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(temp_dir)
```

* Siz `docs.zip` berasiz.
* Skript uni `tmp_docs/` papkaga chiqaradi.
* Endi har bir DOCX fayl alohida mavjud bo‘ladi.

---

### **Step 2 — DOCX faylni o‘qish**

```python
def read_docx(path):
    doc = docx.Document(path)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return [clean_text(p) for p in paras]
```

* Har bir DOCX fayl `docx.Document(path)` orqali o‘qiladi.
* Faqat bo‘sh bo‘lmagan paragraflar olinadi.
* Keyin `clean_text()` orqali tozalanadi:

  * Bo‘sh joylar olib tashlanadi
  * Telefon, email, kredit karta raqami `[PHONE]`, `[EMAIL]`, `[CARD]` bilan almashtiriladi

Natija: har paragraf **tozalangan matn** bo‘ladi.

---

### **Step 3 — Instruction-Response template yaratish**

```python
PROMPT_TEMPLATE = """<|start_of_turn|>user
{instruction}
{input}
<|end_of_turn|>
<|start_of_turn|>assistant
{response}<|end_of_turn|>
"""
```

* Har paragraf `instruction` sifatida olinadi.
* O‘zini javob sifatida ham `response` ga qo‘yadi.
* Shu template orqali model **user input → assistant output** tarzida o‘qitiladi.

Natija: model **har paragrafni akademik javob** sifatida o‘rganadi.

---

### **Step 4 — JSONL fayl hosil qilish**

```python
with open(out_jsonl, "w", encoding="utf-8") as f:
    for e in all_entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
```

* Barcha paragraflar JSONL formatida saqlanadi.
* Har satr quyidagicha bo‘ladi:

```json
{
  "instruction": "Matn paragrafi...",
  "input": "",
  "response": "Matn paragrafi...",
  "template": "<|start_of_turn|>user ... <|end_of_turn|> ..."
}
```

---

### **Step 5 — Tokenization va label tayyorlash (SFT)**

```python
tok_full = tokenizer(full_texts, truncation=True, padding="max_length", max_length=max_length)
tok_prompt = tokenizer(prompts, truncation=True, padding="max_length", max_length=max_length)
```

* Modelga **faqat javob qismi** o‘qitilishi uchun `labels` tayyorlanadi.
* Prompt tokenlari `-100` qilinadi (loss hisoblanmaydi).
* Shunda model **instruction ko‘rsatilib, response yozishni o‘rganadi**.

---

### **Step 6 — Modelni yuklash va LoRA tayyorlash**

```python
model = AutoModelForCausalLM.from_pretrained(...)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_conf)
```

* Model `AutoModelForCausalLM` orqali yuklanadi.
* Agar xohlansa, **4-bit yoki 8-bit** bilan RAM tejamkorligi bilan ishlash mumkin.
* `LoRA` adapter yordamida faqat **kichik qo‘shimcha qatlamlar** o‘rganiladi, asosiy model vaznlari **o‘zgarmaydi**.

Natija: RAM / GPU tejamkor, tezroq fine-tuning.

---

### **Step 7 — Training**

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)
trainer.train()
```

* Trainer JSONL dagi **barcha paragraflarni** tokenlarga aylantirib o‘qitadi.
* Gradient accumulation, fp16, batch size, epochs sozlanadi.
* Har epoch oxirida **LoRA adapter** saqlanadi.

---

### **Step 8 — Natija**

* `train_all.jsonl` → training dataset
* `lora_out/` → tayyor LoRA adapter

Siz boshqa modelga yuklab, **inference** qilishingiz mumkin.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = PeftModel.from_pretrained(base_model, "lora_out")
```

---

## **🔹 Pipeline xulosa**

1. **Zip fayl berasiz** → DOCX lar ochiladi.
2. **Har paragraf tozalanadi** (PII, bo‘sh joy).
3. **Template yaratiladi** (instruction + response).
4. **JSONL hosil qilinadi**.
5. **LoRA fine-tuning** avtomatik bajariladi.
6. Natijada sizda **tayyor adapter** bo‘ladi, asosiy modelni buzmaydi.

---


