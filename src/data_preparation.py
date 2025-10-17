import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Dict, Any

from datasets import load_dataset, DatasetDict
from typing import Dict, Any
from unsloth.chat_templates import get_chat_template, standardize_data_formats


def prepare_dataset(
    dataset_name: str,
    tokenizer,
    num_proc: int = 4,
    system_prompt: str = (
        "You are a helpful, empathetic medical assistant. Provide clear, respectful, "
        "evidence-informed guidance for patient questions. You are not a substitute for a "
        "clinician: avoid making definitive diagnoses or prescribing. Encourage seeking "
        "professional care for personal medical advice."
    ),
) -> DatasetDict:
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name)
    print(f"✅ Loaded dataset '{dataset_name}'. Splits: {list(ds.keys())}")

    def _format_row(row: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]},
        ]
        row["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # SFT: labels are assistant tokens
        )
        return row

    splits = {}
    if "train" in ds:
        splits["train"] = ds["train"].map(
            _format_row,
            num_proc=num_proc,
            remove_columns=[c for c in ds["train"].column_names if c != "text"],
        )
    if "validation" in ds:
        splits["validation"] = ds["validation"].map(
            _format_row,
            num_proc=num_proc,
            remove_columns=[c for c in ds["validation"].column_names if c != "text"],
        )
    if not splits:
        # single-split fallback
        only = ds
        splits["train"] = only.map(
            _format_row,
            num_proc=num_proc,
            remove_columns=[c for c in only.column_names if c != "text"],
        )

    formatted = DatasetDict(splits)
    print(
        "✅ Formatting done. Columns:",
        {k: formatted[k].column_names for k in formatted.keys()},
    )
    return formatted


def analyze_token_lengths(
    formatted: DatasetDict,
    tokenizer,
    field: str = "text",
    split: str = "train",
    num_proc: int = 4,
    plot_bins: int = 100,
    max_items_for_plot: int = None,  # set e.g. 50_000 to cap plotting size if huge
):
    """
    Tokenizes the chosen split's `field` to compute token lengths, prints stats,
    total token count, and plots a histogram.
    """

    ds = formatted[split]

    # Tokenize to lengths only (no truncation so we see true lengths)
    def _len_only(batch):
        enc = tokenizer(batch[field], add_special_tokens=True, truncation=False)
        return {"tok_len": [len(ids) for ids in enc["input_ids"]]}

    ds_with_len = ds.map(_len_only, batched=True, num_proc=num_proc)

    lengths = ds_with_len["tok_len"]
    lengths_np = np.array(lengths, dtype=np.int32)

    total_tokens = int(lengths_np.sum())
    mean_len = float(lengths_np.mean())
    p50 = float(np.percentile(lengths_np, 50))
    p90 = float(np.percentile(lengths_np, 90))
    p95 = float(np.percentile(lengths_np, 95))
    p99 = float(np.percentile(lengths_np, 99))
    max_len = int(lengths_np.max())
    min_len = int(lengths_np.min())

    print("📊 Token stats:")
    print(f"  • Examples:            {len(lengths_np):,}")
    print(f"  • Total tokens:        {total_tokens:,}")
    print(f"  • Mean length:         {mean_len:,.2f}")
    print(f"  • Median (p50):        {p50:,.0f}")
    print(f"  • p90 / p95 / p99:     {p90:,.0f} / {p95:,.0f} / {p99:,.0f}")
    print(f"  • Min / Max:           {min_len} / {max_len}")

    # Optional cap for very large datasets when plotting
    if max_items_for_plot is not None and len(lengths_np) > max_items_for_plot:
        plot_data = lengths_np[:max_items_for_plot]
    else:
        plot_data = lengths_np

    # Histogram (per matplotlib-only rules)
    plt.figure()
    plt.hist(plot_data, bins=plot_bins)
    plt.title(f"Token count distribution ({split})")
    plt.xlabel("Tokens per example")
    plt.ylabel("Frequency")
    plt.show()

    return {
        "total_tokens": total_tokens,
        "mean": mean_len,
        "p50": p50,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "min": min_len,
        "max": max_len,
    }
