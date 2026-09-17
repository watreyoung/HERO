#!/usr/bin/env python3
"""Distributed generation and task metrics for a trained causal HERO model."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from sacrebleu import corpus_bleu
from sacrebleu import sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .causal_data import encode_generation_prompt, target_from_row
except ImportError:  # Direct execution via run_qwen.py.
    from causal_data import encode_generation_prompt, target_from_row


LOGGER = logging.getLogger("hero.evaluate_causal")
_STARTED = time.monotonic()


def stamp(message: str) -> None:
    elapsed = int(time.monotonic() - _STARTED)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now} +{hours:02d}:{minutes:02d}:{seconds:02d}] {message}", flush=True)


LABEL_RE = re.compile(r"\b(true|false)\b", re.IGNORECASE)
ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("rnp", "crcg"))
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument(
        "--tokenizer-name-or-path",
        help="Tokenizer path when evaluating a Trainer checkpoint (defaults to model path)",
    )
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--attn-implementation", default="sdpa", choices=("eager", "sdpa", "flash_attention_2"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Delete prediction shards after writing aggregate metrics",
    )
    return parser.parse_args()


def distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def read_rank_rows(path: str, rank: int, world_size: int):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index % world_size == rank:
                rows.append((index, json.loads(line)))
    return rows


def parse_rnp_label(text: str) -> int | None:
    match = LABEL_RE.search(text)
    if not match:
        return None
    return 1 if match.group(1).lower() == "true" else 0


def edit_similarity(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0
    try:
        import Levenshtein

        distance = Levenshtein.distance(reference, hypothesis)
    except ImportError:
        previous = list(range(len(hypothesis) + 1))
        for row_index, ref_char in enumerate(reference, start=1):
            current = [row_index]
            for col_index, hyp_char in enumerate(hypothesis, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[col_index] + 1,
                        previous[col_index - 1] + (ref_char != hyp_char),
                    )
                )
            previous = current
        distance = previous[-1]
    return 1.0 - distance / max(len(reference), len(hypothesis))


def compute_metrics(task: str, predictions: list[dict]) -> dict[str, float | int]:
    if task == "rnp":
        gold = [int(row["gold"]) for row in predictions]
        parsed = [parse_rnp_label(row["prediction"]) for row in predictions]
        invalid = sum(value is None for value in parsed)
        tp = sum(label == 1 and value == 1 for label, value in zip(gold, parsed))
        fp = sum(label == 0 and value != 0 for label, value in zip(gold, parsed))
        fn = sum(label == 1 and value != 1 for label, value in zip(gold, parsed))
        accuracy = sum(value is not None and value == label for value, label in zip(parsed, gold)) / max(len(gold), 1)
        return {
            "accuracy": round(accuracy, 6),
            "precision": round(tp / max(tp + fp, 1), 6),
            "recall": round(tp / max(tp + fn, 1), 6),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 6),
            "invalid_predictions": invalid,
            "examples": len(gold),
        }

    hypotheses = [row["prediction"].strip() for row in predictions]
    references = [str(row["gold"]).strip() for row in predictions]
    bleu = corpus_bleu(hypotheses, [references], tokenize="13a", use_effective_order=True).score
    sentence_bleu_score = sum(
        sentence_bleu(hypothesis, [reference], tokenize="13a").score
        for hypothesis, reference in zip(hypotheses, references)
    ) / max(len(hypotheses), 1)
    edit_similarity_score = 100.0 * sum(
        edit_similarity(reference, hypothesis)
        for hypothesis, reference in zip(hypotheses, references)
    ) / max(len(hypotheses), 1)
    rouge_l = 100.0 * sum(
        ROUGE_SCORER.score(reference, hypothesis)["rougeL"].fmeasure
        for hypothesis, reference in zip(hypotheses, references)
    ) / max(len(hypotheses), 1)
    exact = 100.0 * sum(h == r for h, r in zip(hypotheses, references)) / max(len(hypotheses), 1)
    return {
        "bleu4": round(bleu, 6),
        "sentence_bleu": round(sentence_bleu_score, 6),
        "rouge_l": round(rouge_l, 6),
        "edit_similarity": round(edit_similarity_score, 6),
        "exact_match": round(exact, 6),
        "examples": len(hypotheses),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    rank, local_rank, world_size = distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("Causal generation evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path or args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        device_map={"": local_rank},
    )
    model.eval()

    ranked_rows = read_rank_rows(args.data_file, rank, world_size)
    total_batches = max(1, (len(ranked_rows) + args.batch_size - 1) // args.batch_size)
    stamp(
        f"[qwen-eval] rank {rank} split {args.split} examples {len(ranked_rows)} "
        f"batches {total_batches} batch_size {args.batch_size} max_new_tokens {args.max_new_tokens}"
    )
    predictions = []
    for batch_index, start in enumerate(range(0, len(ranked_rows), args.batch_size), start=1):
        chunk = ranked_rows[start : start + args.batch_size]
        encoded = [
            encode_generation_prompt(tokenizer, args.task, str(row["code_change"]), args.context_length)
            for _, row in chunk
        ]
        width = max(len(ids) for ids in encoded)
        input_ids = []
        attention_mask = []
        for ids in encoded:
            padding = width - len(ids)
            input_ids.append([tokenizer.pad_token_id] * padding + ids)
            attention_mask.append([0] * padding + [1] * len(ids))
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
        if rank == 0:
            stamp(f"[qwen-eval] {args.split} batch {batch_index}/{total_batches}")
        for (index, row), text in zip(chunk, texts):
            gold = int(row["label"]) if args.task == "rnp" else target_from_row(args.task, row)
            prediction = text.strip()
            predictions.append(
                {
                    "index": index,
                    "gold": gold,
                    "prediction": prediction,
                    "target": gold,
                    "output": prediction,
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"{args.split}_predictions.rank{rank}.jsonl"
    with open(shard_path, "w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        merged = []
        for shard_rank in range(world_size):
            path = output_dir / f"{args.split}_predictions.rank{shard_rank}.jsonl"
            with open(path, encoding="utf-8") as handle:
                merged.extend(json.loads(line) for line in handle if line.strip())
        merged.sort(key=lambda row: row["index"])
        merged_path = output_dir / f"{args.split}_predictions.jsonl"
        with open(merged_path, "w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics = compute_metrics(args.task, merged)
        metrics["examples"] = len(merged)
        metrics_path = output_dir / f"{args.split}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        LOGGER.info("%s metrics: %s", args.split, metrics)
        stamp(f"[qwen-eval] {args.split} metrics: {json.dumps(metrics, sort_keys=True)}")

    if world_size > 1:
        dist.barrier()
    if args.metrics_only:
        if rank == 0:
            (output_dir / f"{args.split}_predictions.jsonl").unlink(missing_ok=True)
            for shard_rank in range(world_size):
                (output_dir / f"{args.split}_predictions.rank{shard_rank}.jsonl").unlink(
                    missing_ok=True
                )
        if world_size > 1:
            dist.barrier()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
