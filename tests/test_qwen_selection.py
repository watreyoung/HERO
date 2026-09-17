import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from qwen.select_and_test import select_and_test
from qwen.evaluate_causal import compute_metrics


class QwenSelectionTests(unittest.TestCase):
    def test_invalid_rnp_output_counts_as_error(self):
        metrics = compute_metrics("rnp", [
            {"gold": 0, "prediction": "unknown"},
            {"gold": 1, "prediction": "true"},
        ])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["f1"], 0.666667)
        self.assertEqual(metrics["invalid_predictions"], 1)

    def test_exact_crcg_comment_has_full_bleu(self):
        metrics = compute_metrics("crcg", [
            {"gold": "Please remove this unused import.",
             "prediction": "Please remove this unused import."},
        ])
        self.assertEqual(metrics["bleu4"], 100.0)

    def test_dev_selects_best_checkpoint_before_single_test(self):
        for task, metric in (("rnp", "f1"), ("crcg", "bleu4")):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                run_dir = Path(directory)
                (run_dir / "tokenizer_config.json").write_text("{}")
                (run_dir / "tokenizer.json").write_text("{}")
                for step in (10, 20, 30):
                    checkpoint = run_dir / f"checkpoint-{step}"
                    checkpoint.mkdir()
                    (checkpoint / "trainer_state.json").write_text(json.dumps({"epoch": step // 10}))
                    (checkpoint / "config.json").write_text("{}")
                    (checkpoint / "model.safetensors").write_text(str(step))

                calls = []

                def fake_run(command, check):
                    self.assertTrue(check)
                    split = command[command.index("--split") + 1]
                    model = Path(command[command.index("--model-name-or-path") + 1])
                    output = Path(command[command.index("--output-dir") + 1])
                    output.mkdir(parents=True, exist_ok=True)
                    calls.append((split, model))
                    if split == "dev":
                        score = {10: 0.4, 20: 0.8, 30: 0.6}[int(model.name.split("-")[-1])]
                        metrics = {metric: score, "examples": 5}
                    else:
                        self.assertEqual(model.name, "best_model")
                        self.assertEqual((model / "model.safetensors").read_text(), "20")
                        metrics = {metric: 0.7, "examples": 5}
                    (output / f"{split}_metrics.json").write_text(json.dumps(metrics))

                args = SimpleNamespace(task=task, run_dir=run_dir, dev_file=run_dir / "dev.json",
                                       test_file=run_dir / "test.json", context_length=512,
                                       max_new_tokens=8, batch_size=2, num_gpus=1,
                                       attn_implementation="sdpa", trust_remote_code=False,
                                       keep_checkpoints=False)
                summary = select_and_test(args, run_command=fake_run)
                self.assertEqual(summary["selected_step"], 20)
                self.assertEqual(summary["selection_scope"], "dev")
                self.assertEqual([split for split, _ in calls], ["dev", "dev", "dev", "test"])
                self.assertFalse(list(run_dir.glob("checkpoint-*")))
                self.assertEqual(len((run_dir / "epoch_metrics.jsonl").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
