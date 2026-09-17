import shlex
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_plm.py"


class LauncherTests(unittest.TestCase):
    def run_launcher(self, task, model, *extra):
        return subprocess.run(
            [sys.executable, str(LAUNCHER), "--task", task, "--model", model,
             "--language", "cpp", "--mode", "original", "--data-root", "/tmp/hero-data",
             "--model-path", "/tmp/hero-model", *extra],
            capture_output=True, text=True, cwd=ROOT,
        )

    def test_all_models_route_to_correct_training_script(self):
        for task in ("RNP", "CRCG"):
            for model in ("roberta", "codebert", "codet5", "codereviewer", "unixcoder"):
                with self.subTest(task=task, model=model):
                    result = self.run_launcher(task, model, "--dry-run")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    cmd = shlex.split(result.stdout)
                    script = "uni_run.py" if model == "unixcoder" else (
                        "run_defect.py" if task == "RNP" and model in ("roberta", "codebert") else "run_gen.py")
                    self.assertEqual(Path(cmd[1]), ROOT / task / script)
                    self.assertIn("--do_train", cmd)
                    self.assertIn("--do_test", cmd)
                    if task == "CRCG" or model != "unixcoder":
                        self.assertIn("--do_eval", cmd)
                    self.assertIn(str(ROOT / "outputs" / task / "cpp" / "original" / model), cmd)

    def test_dry_run_is_side_effect_free(self):
        result = self.run_launcher("RNP", "codebert", "--dry-run", "--no-test",
                                   "--tokenizer-path", "/tmp/roberta-tokenizer")
        self.assertEqual(result.returncode, 0, result.stderr)
        cmd = shlex.split(result.stdout)
        self.assertNotIn("--do_test", cmd)
        self.assertEqual(cmd[cmd.index("--tokenizer_name") + 1], "/tmp/roberta-tokenizer")

    def test_rejects_missing_dataset_before_starting_training(self):
        result = self.run_launcher("RNP", "roberta")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing dataset files", result.stderr)


if __name__ == "__main__":
    unittest.main()
