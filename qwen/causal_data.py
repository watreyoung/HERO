"""Task templates and tokenization for decoder-only code models."""

from __future__ import annotations

from typing import Any


HERO_SPECIAL_TOKENS = [
    "[ADD]",
    "[DELETE]",
    "[URL]",
    "[CC]",
    "[CM]",
    "[HC]",
    "[HISTORY_MSG]",
    "[HISTORY_CODE]",
]

TASK_TEMPLATES = {
    "rnp": (
        "Predict whether the code change needs a review comment. "
        "Answer with exactly true or false.\n\nCode change:\n",
        "\n\nAnswer:\n",
    ),
    "crcg": (
        "Write a concise and actionable code review comment for the code change.\n\n"
        "Code change:\n",
        "\n\nReview comment:\n",
    ),
}


def target_from_row(task: str, row: dict[str, Any]) -> str:
    if task == "rnp":
        return "true" if int(row["label"]) == 1 else "false"
    if task == "crcg":
        return str(row["review_comment"])
    raise ValueError(f"Unsupported task: {task}")


def prompt_from_source(task: str, source: str) -> str:
    prefix, suffix = TASK_TEMPLATES[task]
    return prefix + source.strip() + suffix


def encode_supervised_example(
    tokenizer,
    task: str,
    source: str,
    target: str,
    context_length: int,
    max_target_length: int,
) -> dict[str, list[int]]:
    """Encode one example and mask prompt tokens from the language-model loss.

    HERO puts the current change at the right edge of packed inputs. When a
    model's tokenizer uses more pieces than the dataset packer, trimming from
    the left drops the oldest history while retaining that current change.
    """
    prefix, suffix = TASK_TEMPLATES[task]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    source_ids = tokenizer.encode(source.strip(), add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    target_ids = tokenizer.encode(target.strip(), add_special_tokens=False)
    target_ids = target_ids[: max(max_target_length - 1, 0)]

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("The tokenizer must define eos_token_id")
    target_ids = target_ids + [eos_id]

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    source_ids = source_ids[-context_length:]
    prompt_ids = prefix_ids + source_ids + suffix_ids
    input_ids = prompt_ids + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def encode_generation_prompt(tokenizer, task: str, source: str, source_length: int) -> list[int]:
    prefix, suffix = TASK_TEMPLATES[task]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    source_ids = tokenizer.encode(source.strip(), add_special_tokens=False)
    if source_length <= 0:
        raise ValueError("source_length must be positive")
    return prefix_ids + source_ids[-source_length:] + suffix_ids


class CausalDataCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        import torch

        longest = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            longest = ((longest + multiple - 1) // multiple) * multiple

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = longest - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
