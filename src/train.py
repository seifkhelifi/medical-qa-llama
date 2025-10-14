from unsloth import FastLanguageModel
import torch
import os

from trl import SFTTrainer, SFTConfig, setup_chat_format
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

import wandb

from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

from dotenv import load_dotenv

from data_preparation import prepare_llama3_dataset


import os

# BEFORE loading the model and OUTSIDE is_main_process():
user_secrets = UserSecretsClient()
hf_token = os.getenv("HF_TOKEN") or user_secrets.get_secret("HF_TOKEN")
assert hf_token, "Set HF_TOKEN in Kaggle Secrets"
os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token   # ensures no prompt on any rank
login(hf_token)  # idempotent; safe on all ranks

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

def is_main_process() -> bool:
    r = os.environ.get("RANK")
    return r is None or r == "0"

# Optional: login/init only on rank-0
if is_main_process():
    wb_token = os.getenv("WANDB_TOKEN") or UserSecretsClient().get_secret("WANDB_API_KEY")
    import wandb
    wandb.login(key=wb_token)
    wandb.init(project="Fine-tune Llama-3.1-8B for medical qa", job_type="training", anonymous="allow")
    

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length = 1024,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,   # LoRA path (matches your get_peft_model below)
    device_map = None,         # <-- IMPORTANT for DDP
)


tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

formatted_dataset = prepare_llama3_dataset(
    dataset_name="petkopetkov/medical-question-answering-synthetic",
    tokenizer=tokenizer
)

# Check result
print(formatted_dataset)
print(formatted_dataset['train']['text'][2])

model = FastLanguageModel.get_peft_model(
    model,
    r=16, 
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
    lora_dropout=0, 
    bias="none", 
   
    use_gradient_checkpointing="unsloth", 
    random_state=3407,
    use_rslora=False, 
    loftq_config=None,
)

report_to = ["wandb"] if is_main_process() else ["none"]  # <- key change


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted_dataset["train"],
    eval_dataset = formatted_dataset["validation"],
    dataset_text_field = "text",
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,     # ↓ see notes on memory below
        gradient_accumulation_steps = 8,     # ↑ keeps global batch ~same as before
        warmup_steps = 50,
        max_steps = 200,                     # do more than 60 steps for signal
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",

        # >>> DDP/Unsloth important:
        ddp_find_unused_parameters = False,

        # >>> Logging:
        report_to = ["wandb"],               # Trainer will push to your W&B project
        run_name = "llama-8b-ddp-medqa",

        # (Optional niceties)
        save_steps = 200,
        save_total_limit = 2,
    ),
    # Optional: better context packing utilization (if your data are short)
    # packing=True,
)



trainer_stats = trainer.train()