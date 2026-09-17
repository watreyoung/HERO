#!/usr/bin/env python3
"""Select a Qwen checkpoint on dev task metrics, then evaluate it on test."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
WEIGHT_PATTERNS = ("model*.safetensors", "pytorch_model*.bin")
MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors.index.json",
               "pytorch_model.bin.index.json")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
                   "chat_template.jinja")


def log(message: str) -> None:
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [qwen] {message}", flush=True)


def checkpoints(run_dir: Path) -> list[tuple[int, int, Path]]:
    result = []
    for path in run_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        state_file = path / "trainer_state.json"
        if not state_file.is_file():
            continue
        state = json.loads(state_file.read_text(encoding="utf-8"))
        epoch = max(1, int(round(float(state.get("epoch", 0)))))
        result.append((epoch, step, path))
    return sorted(result, key=lambda item: item[1])


def evaluation_command(args, model_path: Path, data_file: Path, output: Path, split: str) -> list[str]:
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={args.num_gpus}", str(HERE / "evaluate_causal.py"),
               "--task", args.task, "--model-name-or-path", str(model_path),
               "--tokenizer-name-or-path", str(args.run_dir if split == "dev" else model_path),
               "--data-file", str(data_file), "--output-dir", str(output), "--split", split,
               "--context-length", str(args.context_length),
               "--max-new-tokens", str(args.max_new_tokens), "--batch-size", str(args.batch_size),
               "--attn-implementation", args.attn_implementation]
    if split == "dev":
        command.append("--metrics-only")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def primary_score(task: str, metrics: dict) -> float:
    metric = "f1" if task == "rnp" else "bleu4"
    if metric not in metrics or not math.isfinite(float(metrics[metric])):
        raise ValueError(f"Invalid dev {metric}: {metrics}")
    if not metrics.get("examples"):
        raise ValueError("Dev metrics contain no examples")
    return float(metrics[metric])


def export_best(checkpoint: Path, run_dir: Path) -> Path:
    weights = [path for pattern in WEIGHT_PATTERNS for path in checkpoint.glob(pattern)]
    if not weights:
        raise RuntimeError(f"Checkpoint has no consolidated model weights: {checkpoint}")
    if not (checkpoint / "config.json").is_file():
        raise RuntimeError(f"Checkpoint has no model config: {checkpoint}")
    if not (run_dir / "tokenizer_config.json").is_file():
        raise RuntimeError(f"Tokenizer was not saved in {run_dir}")
    if not any((run_dir / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise RuntimeError(f"Tokenizer vocabulary was not saved in {run_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=".best_model-", dir=run_dir))
    try:
        files = weights + [checkpoint / name for name in MODEL_FILES if (checkpoint / name).is_file()]
        files += [run_dir / name for name in TOKENIZER_FILES if (run_dir / name).is_file()]
        for source in files:
            shutil.copy2(source, temporary / source.name)
        destination = run_dir / "best_model"
        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def select_and_test(args, run_command=subprocess.run) -> dict:
    candidates = checkpoints(args.run_dir)
    if not candidates:
        raise RuntimeError(f"No epoch checkpoints in {args.run_dir}")
    history = []
    for epoch, step, checkpoint in candidates:
        output = args.run_dir / "epoch_evaluation" / f"epoch_{epoch}_step_{step}"
        log(f"evaluating dev: epoch={epoch} step={step}")
        run_command(evaluation_command(args, checkpoint, args.dev_file, output, "dev"), check=True)
        metrics = json.loads((output / "dev_metrics.json").read_text(encoding="utf-8"))
        score = primary_score(args.task, metrics)
        history.append({"epoch": epoch, "step": step, "checkpoint": str(checkpoint),
                        "dev": metrics, "selection_score": score})
        log(f"dev {'f1' if args.task == 'rnp' else 'bleu4'}={score:.6f}")

    selected = max(history, key=lambda row: (row["selection_score"], -row["step"]))
    checkpoint = Path(selected["checkpoint"])
    best_model = export_best(checkpoint, args.run_dir)
    evaluation_dir = args.run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "dev_metrics.json").write_text(
        json.dumps(selected["dev"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    history_file = args.run_dir / "epoch_metrics.jsonl"
    with history_file.open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    log(f"selected epoch={selected['epoch']} step={selected['step']} from dev")
    run_command(evaluation_command(args, best_model, args.test_file, evaluation_dir, "test"), check=True)
    test = json.loads((evaluation_dir / "test_metrics.json").read_text(encoding="utf-8"))
    summary = {"task": args.task, "selection_scope": "dev",
               "selection_metric": "f1" if args.task == "rnp" else "bleu4",
               "selected_epoch": selected["epoch"], "selected_step": selected["step"],
               "selected_source_checkpoint": str(checkpoint), "best_model": str(best_model),
               "dev_metrics": selected["dev"], "test_metrics": test}
    (args.run_dir / "selection.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.keep_checkpoints:
        for _, _, path in candidates:
            shutil.rmtree(path)
    log(f"test metrics: {json.dumps(test, sort_keys=True)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("rnp", "crcg"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--attn-implementation", default="sdpa",
                        choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()
    select_and_test(args)


if __name__ == "__main__":
    main()
