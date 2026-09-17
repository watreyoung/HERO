#!/usr/bin/env python3
"""Train Qwen3-4B for HERO, select on dev, and test the selected model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
QWEN = ROOT / "qwen"
ALIASES = {"all": "hero", "cc": "history", "msg": "message"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("rnp", "crcg"))
    parser.add_argument("--language", required=True, choices=("java", "cpp"))
    parser.add_argument("--variant", default="hero", choices=(
        "original", "hero", "history", "message", "all", "cc", "msg", "last1", "naive", "random"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--dev-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "qwen")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--max-target-length", type=int)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpus", help="Comma-separated visible GPU IDs")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--preprocessing-workers", type=int, default=4)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--deepspeed-config", type=Path, default=QWEN / "deepspeed_zero3_bf16.json")
    parser.add_argument("--no-deepspeed", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa",
                        choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--selection-only", action="store_true",
                        help="Resume dev selection and test using saved epoch checkpoints")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("context_length", "num_gpus", "train_batch_size", "eval_batch_size",
                 "gradient_accumulation_steps", "epochs", "logging_steps",
                 "preprocessing_workers", "dataloader_workers", "learning_rate"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_target_length is None:
        args.max_target_length = 8 if args.task == "rnp" else 256
    if args.max_target_length <= 0 or args.max_target_length >= args.context_length:
        parser.error("--max-target-length must be positive and below --context-length")
    if args.gpus:
        gpu_ids = [value.strip() for value in args.gpus.split(",")]
        if len(gpu_ids) != args.num_gpus or any(not value.isdigit() for value in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
            parser.error("--gpus must contain --num-gpus distinct numeric IDs")
        args.gpus = ",".join(gpu_ids)
    args.variant = ALIASES.get(args.variant, args.variant)
    args.output_dir = (args.output_dir or args.output_root / args.task / args.language /
                       args.variant / f"ctx{args.context_length}").expanduser().resolve()
    args.cache_dir = (args.cache_dir or args.output_dir / "cache").expanduser().resolve()
    data_dir = args.data_root.expanduser().resolve() / args.language / args.task.upper() / args.variant
    args.train_file = (args.train_file or data_dir / "train.json").expanduser().resolve()
    args.dev_file = (args.dev_file or data_dir / "dev.json").expanduser().resolve()
    args.test_file = (args.test_file or data_dir / "test.json").expanduser().resolve()
    return args


def commands(args):
    train = [sys.executable, "-m", "torch.distributed.run", "--standalone",
             f"--nproc_per_node={args.num_gpus}", str(QWEN / "train_causal.py"),
             "--task", args.task, "--model-name-or-path", args.model_name_or_path,
             "--train-file", str(args.train_file), "--validation-file", str(args.dev_file),
             "--output-dir", str(args.output_dir), "--cache-dir", str(args.cache_dir),
             "--context-length", str(args.context_length),
             "--max-target-length", str(args.max_target_length),
             "--train-batch-size", str(args.train_batch_size),
             "--eval-batch-size", str(args.eval_batch_size),
             "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
             "--learning-rate", str(args.learning_rate), "--num-train-epochs", str(args.epochs),
             "--max-steps", str(args.max_steps), "--warmup-ratio", str(args.warmup_ratio),
             "--logging-steps", str(args.logging_steps),
             "--preprocessing-workers", str(args.preprocessing_workers),
             "--dataloader-workers", str(args.dataloader_workers),
             "--attn-implementation", args.attn_implementation]
    if not args.no_deepspeed:
        train.extend(("--deepspeed", str(args.deepspeed_config)))
    if args.resume_from_checkpoint:
        train.extend(("--resume-from-checkpoint", args.resume_from_checkpoint))
    if args.trust_remote_code:
        train.append("--trust-remote-code")

    select = [sys.executable, str(QWEN / "select_and_test.py"), "--task", args.task,
              "--run-dir", str(args.output_dir), "--dev-file", str(args.dev_file),
              "--test-file", str(args.test_file), "--context-length", str(args.context_length),
              "--max-new-tokens", str(args.max_target_length),
              "--batch-size", str(args.eval_batch_size), "--num-gpus", str(args.num_gpus),
              "--attn-implementation", args.attn_implementation]
    if args.trust_remote_code:
        select.append("--trust-remote-code")
    if args.keep_checkpoints:
        select.append("--keep-checkpoints")
    return train, select


def run_logged(command, log_path, env):
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)


def main(argv=None):
    args = parse_args(argv)
    train, select = commands(args)
    env = os.environ.copy()
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    stages = [("train", train), ("selection", select)]
    if args.selection_only:
        stages = stages[1:]
    for name, command in stages:
        print(f"{name}: {shlex.join(command)}", flush=True)
    if args.dry_run:
        return
    required = [args.dev_file, args.test_file]
    if not args.selection_only:
        required.append(args.train_file)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing dataset files: " + ", ".join(missing))
    if not args.no_deepspeed and not args.selection_only and not args.deepspeed_config.is_file():
        raise SystemExit(f"Missing DeepSpeed config: {args.deepspeed_config}")
    if not args.selection_only and not args.resume_from_checkpoint and any(args.output_dir.glob("checkpoint-*")):
        raise SystemExit("Output already has epoch checkpoints; use --selection-only, --resume-from-checkpoint, or a new --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for name, command in stages:
        run_logged(command, args.output_dir / f"{name}.log", env)


if __name__ == "__main__":
    main()
