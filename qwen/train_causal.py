#!/usr/bin/env python3
"""Distributed full fine-tuning for HERO's decoder-only model family."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

try:
    from .causal_data import CausalDataCollator, HERO_SPECIAL_TOKENS, encode_supervised_example, target_from_row
except ImportError:  # Direct execution via run_qwen.py.
    from causal_data import CausalDataCollator, HERO_SPECIAL_TOKENS, encode_supervised_example, target_from_row


LOGGER = logging.getLogger("hero.train_causal")
_STARTED = time.monotonic()


def stamp(message: str) -> None:
    """Print a stage line with wall time and elapsed time since this process started."""
    elapsed = int(time.monotonic() - _STARTED)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now} +{hours:02d}:{minutes:02d}:{seconds:02d}] {message}", flush=True)


def _rank0() -> bool:
    return os.environ.get("RANK", "0") in {"0", ""}


class StagePrinter(TrainerCallback):
    """Emit progress for long-running training and checkpoint stages."""

    def on_train_begin(self, args, state, control, **kwargs):
        if not _rank0():
            return
        stamp(
            f"[qwen] train begin epochs {args.num_train_epochs} "
            f"max_steps {state.max_steps} logging_steps {args.logging_steps}"
        )

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not _rank0():
            return
        stamp(f"[qwen] epoch {int(state.epoch or 0) + 1} begin step {state.global_step}")

    def on_epoch_end(self, args, state, control, **kwargs):
        if not _rank0():
            return
        stamp(f"[qwen] epoch {state.epoch} end step {state.global_step}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not _rank0():
            return
        loss = None if metrics is None else metrics.get("eval_loss")
        stamp(f"[qwen] loss-eval end step {state.global_step} epoch {state.epoch} eval_loss {loss}")

    def on_save(self, args, state, control, **kwargs):
        if not _rank0():
            return
        stamp(f"[qwen] epoch checkpoint saved step {state.global_step}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("rnp", "crcg"))
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-target-length", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--eval-batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--preprocessing-workers", type=int, default=4)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--deepspeed")
    parser.add_argument("--attn-implementation", default="sdpa", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    set_seed(args.seed)
    has_validation = bool(args.validation_file)

    training_kwargs = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=has_validation,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_strategy="epoch",
        eval_strategy="epoch" if has_validation else "no",
        save_total_limit=None,
        load_best_model_at_end=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        report_to="none",
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    # transformers renamed evaluation_strategy to eval_strategy. Supporting
    # both keeps remote environments with a slightly older release usable.
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in parameters:
        training_kwargs["evaluation_strategy"] = training_kwargs.pop("eval_strategy")
    # Construct this before the model. For ZeRO-3, TrainingArguments installs
    # the integration that partitions parameters during from_pretrained().
    training_args = TrainingArguments(**training_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError(f"Tokenizer {args.model_name_or_path!r} has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": HERO_SPECIAL_TOKENS})

    data_files = {"train": args.train_file}
    if args.validation_file:
        data_files["validation"] = args.validation_file

    def tokenize(row):
        target = target_from_row(args.task, row)
        return encode_supervised_example(
            tokenizer=tokenizer,
            task=args.task,
            source=str(row["code_change"]),
            target=target,
            context_length=args.context_length,
            max_target_length=args.max_target_length,
        )

    with training_args.main_process_first(desc="building tokenized dataset cache"):
        datasets = load_dataset("json", data_files=data_files, cache_dir=args.cache_dir)
        tokenized = datasets.map(
            tokenize,
            remove_columns=datasets["train"].column_names,
            num_proc=max(args.preprocessing_workers, 1),
            desc="Tokenizing HERO data",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    if n_added:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if _rank0():
        stamp(
            f"[qwen] rank 0 examples train {len(tokenized['train'])} "
            f"dev {len(tokenized['validation']) if has_validation else 0} "
            f"context {args.context_length} per_device_batch {args.train_batch_size}"
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=CausalDataCollator(tokenizer.pad_token_id),
        callbacks=[StagePrinter()],
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    rank = os.environ.get("RANK", "0")
    stamp(f"[qwen] rank {rank} training complete; dev task metrics select the saved checkpoint")
    if trainer.is_world_process_zero():
        stamp("[qwen] rank 0 saving tokenizer")
        tokenizer.save_pretrained(args.output_dir)
        stamp("[qwen] rank 0 tokenizer saved")
        metrics = dict(train_result.metrics)
        metrics["train_examples"] = len(tokenized["train"])
        metrics["context_length"] = args.context_length
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        stamp("[qwen] rank 0 saving trainer state")
        trainer.save_state()
        stamp("[qwen] rank 0 trainer state saved")
        run_config = vars(args).copy()
        run_config["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
        Path(args.output_dir, "hero_run_config.json").write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stamp("[qwen] rank 0 train bookkeeping done")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    stamp(f"[qwen] rank {rank} train process done")


if __name__ == "__main__":
    main()
