from datasets import load_dataset, DatasetDict

def prepare_llama3_dataset(dataset_name: str, tokenizer, num_proc: int = 4) -> DatasetDict:
    """
    Loads and formats a dataset using the Llama 3 chat template.

    Args:
        dataset_name (str): The name or path of the dataset to load.
        tokenizer: The tokenizer with a valid Llama 3 chat template.
        num_proc (int): Number of processes to use when mapping (default = 4).

    Returns:
        DatasetDict: The formatted dataset ready for training.
    """

    llama3_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"""
    tokenizer.chat_template = llama3_template

    dataset = load_dataset(dataset_name)
    print(f"Dataset '{dataset_name}' loaded successfully.")

    def format_chat_template(row):
        instruction = (
            "You are a helpful and empathetic medical assistant. "
            "Provide clear, respectful, and informative responses to patient questions. "
            "Do not diagnose or prescribe—always suggest seeing a healthcare provider for personal medical advice."
        )

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]}
        ]

        row["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return row

    formatted_dataset = DatasetDict({
        "train": dataset["train"].map(
            format_chat_template,
            num_proc=num_proc,
            remove_columns=["input", "output"]
        ),
        "validation": dataset["validation"].map(
            format_chat_template,
            num_proc=num_proc,
            remove_columns=["input", "output"]
        )
    })

    print("✅ Dataset formatting completed.")
    return formatted_dataset
