# ==========================================================
# Fine-tune Llama 3.1 8B with Unsloth + DDP + W&B
# Author: you
# ==========================================================

import os
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from kaggle_secrets import UserSecretsClient
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from data_preparation import prepare_dataset

# ==========================================================
# Setup Environment
# ==========================================================
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_DISABLE_CACHE"] = "1"
os.environ["UNSLOTH_DISABLE_RL_PATCH"] = "1"


def is_main_process() -> bool:
    r = os.environ.get("RANK")
    return r is None or r == "0"


# Set correct GPU per process
if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

# ==========================================================
# Hugging Face Authentication (for all ranks)
# ==========================================================
user_secrets = UserSecretsClient()
hf_token = os.getenv("HF_TOKEN")

# ==========================================================
# Optional: W&B on main process only
# ==========================================================
if is_main_process():
    import wandb

    user_secrets = UserSecretsClient()
    wb_token = user_secrets.get_secret("WANDB_API_KEY")
    if wb_token:
        wandb.login(key=wb_token)
        wandb.init(
            project="SFT-medical-QA",
            job_type="training",
            anonymous="allow",
        )
    else:
        print("⚠️ No W&B token found. Training will continue without logging.")


# ==========================================================
# Load Model
# ==========================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-bnb-4bit",
    max_seq_length=1024,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,  # LoRA finetuning mode
    device_map=None,  # DDP needs full copy per rank
    token=hf_token,
)


tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================================
# Prepare Dataset
# ==========================================================

formatted_dataset = prepare_dataset(
    dataset_name="petkopetkov/medical-question-answering-synthetic", tokenizer=tokenizer
)


# IMPORTANT: must match how your chat_template renders the start of the assistant turn.
# With your template, each assistant block begins with this *exact* substring + newline:
response_template = "<|im_start|>assistant\n"

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

if is_main_process():
    print("✅ Dataset formatting completed.")
    print(formatted_dataset["train"][0]["text"][:500])

# ==========================================================
# Apply LoRA (PEFT)
# ==========================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)


# ==========================================================
# Trainer Configuration
# ==========================================================
report_to = ["wandb"] if is_main_process() else ["none"]

training_args = TrainingArguments(
    per_device_train_batch_size=16,
    # gradient_accumulation_steps=4,
    warmup_steps=50,
    max_steps=101,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    eval_strategy="steps",  # ← add this
    eval_steps=50,  # evaluate every 50 steps (or 100 for cheaper)    optim="adamw_8bit",
    optim="adamw_8bit",  # Faster and more stable
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    ddp_find_unused_parameters=False,
    report_to=report_to,
    run_name="llama-8b-ddp-medqa",
    save_steps=200,
    save_total_limit=2,
    average_tokens_across_devices=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["validation"],
    dataset_text_field="text",
    max_seq_length=1024,
    dataset_num_proc=2,
    args=training_args,
    data_collator=collator,
)

# ==========================================================
# Train
# ==========================================================
if is_main_process():
    print("🚀 Starting DDP Training ...")

trainer_stats = trainer.train()

if is_main_process():
    print("✅ Training completed successfully!")
    print(trainer_stats)
