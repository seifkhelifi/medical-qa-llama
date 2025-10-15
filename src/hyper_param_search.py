# ==========================================================
# Fine-tune Llama 3.1 8B with Unsloth + DDP + W&B (sweep-ready)
# ==========================================================
import os
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from kaggle_secrets import UserSecretsClient
import numpy as np

from data_preparation import prepare_llama3_dataset

# --------------------------
# DDP env
# --------------------------
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def is_main_process() -> bool:
    r = os.environ.get("RANK")
    return r is None or r == "0"

if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

# --------------------------
# Secrets & tokens
# --------------------------
user_secrets = UserSecretsClient()
hf_token = os.getenv("HF_TOKEN")

# --------------------------
# Optional: W&B (rank 0 only)
# --------------------------
wandb = None
if is_main_process():
    import wandb as _wandb
    wandb = _wandb
    wb_token = user_secrets.get_secret("WANDB_API_KEY")
    if wb_token:
        wandb.login(key=wb_token)
        # Don't init here; init per-trial (sweep agent does it automatically)
    else:
        print("⚠️ No W&B token found. Training will continue without logging.")

# --------------------------
# ONE-TIME: load tokenizer & dataset (shared across trials)
# --------------------------
# We'll load tokenizer once at max length=1024 and later re-create model per trial
# (tokenizer ignores max length at runtime; model must be rebuilt per seq length)
base_model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

_, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=1024,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
    device_map=None,
    token=hf_token,
)
base_tokenizer.padding_side = "right"
if base_tokenizer.pad_token is None:
    base_tokenizer.pad_token = base_tokenizer.eos_token

formatted_dataset = prepare_llama3_dataset(
    dataset_name="petkopetkov/medical-question-answering-all",
    tokenizer=base_tokenizer
)
if is_main_process():
    print("✅ Dataset formatting completed.")

# --------------------------
# Trial runner (one sweep trial)
# --------------------------
def run_trial():
    # Rank 0 pulls hyperparams from wandb.config; other ranks mirror via env fallbacks
    cfg = {
        # defaults (also used if wandb is None)
        "learning_rate": float(os.environ.get("LR", 2e-4)),
        "warmup_ratio": float(os.environ.get("WARMUP_RATIO", 0.03)),
        "weight_decay": float(os.environ.get("WEIGHT_DECAY", 0.01)),
        "lr_scheduler_type": os.environ.get("SCHEDULER", "linear"),
        "lora_r": int(os.environ.get("LORA_R", 8)),
        "lora_alpha": int(os.environ.get("LORA_ALPHA", 16)),
        "lora_dropout": float(os.environ.get("LORA_DROPOUT", 0.0)),
        "max_seq_length": int(os.environ.get("MAX_SEQ_LEN", 512)),
        "per_device_train_batch_size": int(os.environ.get("BATCH", 8)),
        "gradient_accumulation_steps": int(os.environ.get("GAS", 4)),
        "max_steps": int(os.environ.get("MAX_STEPS", 40)),
        "eval_every": int(os.environ.get("EVAL_EVERY", 40)),
    }
    if wandb is not None and wandb.run is not None:
        # Overwrite from sweep config
        for k in cfg:
            if k in wandb.config:
                cfg[k] = wandb.config[k]

    # --------------------------
    # Build model fresh per trial (seq length matters for speed)
    # --------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        device_map=None,
        token=hf_token,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # --------------------------
    # Training args (proxy fast by default)
    # --------------------------
    # Compute warmup steps from ratio (simple & robust in sweeps)
    warmup_steps = int(cfg["max_steps"] * cfg["warmup_ratio"])

    report_to = ["wandb"] if (is_main_process() and (wandb is not None)) else ["none"]

    training_args = TrainingArguments(
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_steps=warmup_steps,
        max_steps=cfg["max_steps"],
        learning_rate=cfg["learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=cfg["eval_every"],        # log at end of proxy
        optim="adamw_8bit",
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        seed=3407,
        output_dir="outputs",
        ddp_find_unused_parameters=False,
        report_to=report_to,
        run_name="llama-8b-ddp-medqa",
        save_strategy="no",                     # no ckpt in proxy
        eval_strategy="steps",
        eval_steps=cfg["eval_every"],           # 1 eval at end
        save_total_limit=1,
        average_tokens_across_devices=False,    # avoids fused loss in-place issue
    )

    # --------------------------
    # Trainer (enable packing for speed)
    # --------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["validation"],
        dataset_text_field="text",
        dataset_num_proc=2,
        packing=True,
        args=training_args,
        group_by_length=True,
    )

    if is_main_process():
        print("🚀 Starting trial with config:", cfg)

    # Train
    trainer_stats = trainer.train()

    # Quick proxy metric: last eval loss if present; else training loss
    proxy_val = None
    if trainer.state.log_history:
        # find last eval loss
        eval_losses = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]
        if len(eval_losses) > 0:
            proxy_val = float(eval_losses[-1])

    if proxy_val is None and trainer_stats is not None:
        try:
            proxy_val = float(trainer_stats.training_loss)
        except Exception:
            proxy_val = np.nan

    if is_main_process():
        print(f"✅ Trial done. proxy_val_loss = {proxy_val}")

    if is_main_process() and wandb is not None and wandb.run is not None:
        wandb.log({"proxy_val_loss": proxy_val})

    # Barrier so all ranks finish before next trial
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

# --------------------------
# Entry point: run one sweep trial
# --------------------------
if __name__ == "__main__":
    # If a sweep/agent is driving, rank 0 will have a wandb run already.
    if is_main_process() and (wandb is not None):
        if wandb.run is None:
            # not a sweep? allow manual single-run init
            wandb.init(project="Fine-tune Llama-3.1-8B for medical QA", job_type="training", anonymous="allow")

    run_trial()

    if is_main_process() and (wandb is not None) and (wandb.run is not None):
        wandb.finish()
