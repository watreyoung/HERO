#!/usr/bin/env python3
"""Launch one RNP or CRCG experiment with any supported PLM."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MODELS = ("roberta", "codebert", "codet5", "codereviewer", "unixcoder")


def command(args):
    task_dir = ROOT / args.task
    data_dir = Path(args.data_root).expanduser().resolve()
    model_path = str(Path(args.model_path).expanduser().resolve())
    output_dir = Path(args.output_root).expanduser().resolve() / args.task / args.language / args.mode / args.model
    files = [data_dir / args.language / args.task / args.mode / f"{split}.json"
             for split in ("train", "dev", "test")]

    if args.model == "unixcoder":
        script = "uni_run.py"
        cmd = [sys.executable, str(task_dir / script), "--model_name_or_path", model_path,
               "--output_dir", str(output_dir)]
        names = ("train_data_file", "eval_data_file", "test_data_file") if args.task == "RNP" else (
            "train_filename", "dev_filename", "test_filename")
        for name, path in zip(names, files):
            cmd.extend((f"--{name}", str(path)))
        cmd.extend(("--num_train_epochs", str(args.epochs or (5 if args.task == "RNP" else 15)),
                    "--train_batch_size", str(args.batch_size or (16 if args.task == "RNP" else 32)),
                    "--eval_batch_size", str(args.eval_batch_size or (32 if args.task == "RNP" else 16)),
                    "--learning_rate", str(args.learning_rate or (2e-5 if args.task == "RNP" else 5e-5)),
                    "--seed", str(args.seed)))
        if args.task == "RNP":
            cmd.extend(("--block_size", str(args.source_length), "--cpu_cont", str(args.workers)))
        else:
            cmd.extend(("--max_source_length", str(args.source_length),
                        "--max_target_length", str(args.target_length or 150),
                        "--beam_size", str(args.beam_size)))
            if args.train:
                cmd.append("--do_eval")
    else:
        encoder = args.model in ("roberta", "codebert")
        script = "run_defect.py" if args.task == "RNP" and encoder else "run_gen.py"
        model_type = "roberta" if encoder else "codet5"
        batch_size = args.batch_size or (64 if encoder else 16)
        learning_rate = args.learning_rate or (5e-5 if args.task == "RNP" else
            (1e-5 if args.mode == "original" or encoder else 1e-4))
        cmd = [sys.executable, str(task_dir / script),
               "--task", args.mode, "--sub_task", args.language,
               "--model_type", model_type, "--model_name_or_path", model_path,
               "--tokenizer_name", args.tokenizer_path or model_path,
               "--data_dir", str(data_dir), "--output_dir", str(output_dir),
               "--cache_path", str(output_dir / "cache"),
               "--summary_dir", str(output_dir / "tensorboard"),
               "--res_dir", str(output_dir / "predictions"),
               "--num_train_epochs", str(args.epochs or (10 if args.task == "RNP" else 30)),
               "--train_batch_size", str(batch_size),
               "--eval_batch_size", str(args.eval_batch_size or batch_size),
               "--learning_rate", str(learning_rate),
               "--max_source_length", str(args.source_length),
               "--max_target_length", str(args.target_length or (3 if args.task == "RNP" else 150)),
               "--patience", str(args.patience), "--beam_size", str(args.beam_size),
               "--seed", str(args.seed)]
        if args.train:
            cmd.extend(("--do_eval", "--save_last_checkpoints", "--always_save_model"))
            if not encoder or args.task == "CRCG":
                cmd.append("--do_eval_bleu")
    if args.train:
        cmd.append("--do_train")
    if args.test:
        cmd.append("--do_test")
    return cmd, task_dir, output_dir, files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("RNP", "CRCG"), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--language", choices=("cpp", "java"), required=True)
    parser.add_argument("--mode", choices=("original", "cc", "msg", "all"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", help="Defaults to --model-path")
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--source-length", type=int, default=512)
    parser.add_argument("--target-length", type=int)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4, help="UniXcoder RNP worker count")
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.model == "unixcoder" and args.tokenizer_path:
        parser.error("--tokenizer-path is unsupported for unixcoder")
    if not args.train and not args.test:
        parser.error("enable --train or --test")
    for name in ("epochs", "batch_size", "eval_batch_size", "learning_rate", "source_length",
                 "target_length", "patience", "beam_size", "workers"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    cmd, task_dir, output_dir, files = command(args)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    required = files if args.train and args.test else files[:2] if args.train else files[2:]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing dataset files: " + ", ".join(missing))
    if args.test and not args.train and not list(output_dir.glob("checkpoint-*/pytorch_model.bin")):
        parser.error(f"no checkpoints found in {output_dir}; train first or set --output-root")
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ("cache", "tensorboard", "predictions"):
        (output_dir / child).mkdir(exist_ok=True)
    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    return subprocess.run(cmd, cwd=task_dir, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
