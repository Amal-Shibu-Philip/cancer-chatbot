import os
os.environ["OMP_NUM_THREADS"] = "8"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch

BASE_MODEL  = "EleutherAI/gpt-neo-1.3B"
TRAIN_FILE  = "train.jsonl"
VAL_FILE    = "val.jsonl"
OUTPUT_DIR  = "qlora-neo1.3b-lungcancer"
MAX_LEN     = 256
BATCH_SIZE  = 4
EPOCHS      = 3

def tokenize_fn(examples):
    out = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LEN)
    out["labels"] = out["input_ids"].copy()
    return out

if __name__ == "__main__":
    print(" Loading dataset")
    ds = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

    print(" Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    print(" Tokenizing")
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    print(" Loading 4-bit NF4 model on CPU")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config, device_map={"": "cpu"}, low_cpu_mem_usage=True)

    print(" Preparing for k-bit training")
    model = prepare_model_for_kbit_training(model)

    print(" Injecting LoRA adapters")
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["c_fc", "c_proj"], lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_cfg)

    print(" Setting up TrainingArguments")
    training_args = TrainingArguments(output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE,
                                      gradient_accumulation_steps=1, learning_rate=3e-4, logging_steps=20, save_steps=100,
                                      save_total_limit=2, gradient_checkpointing=False, dataloader_num_workers=4, fp16=False, push_to_hub=False)

    print(" Initializing Trainer")
    trainer = Trainer(model=model, args=training_args, train_dataset=ds["train"], eval_dataset=ds["validation"], tokenizer=tokenizer)

    print(" Starting 4-bit NF4 LoRA fine-tuning")
    trainer.train()

    print(" Done—saving model")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f" Saved fine-tuned model and tokenizer to '{OUTPUT_DIR}'")
