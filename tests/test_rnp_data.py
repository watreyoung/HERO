import json
import tempfile
import unittest
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RNP"))
from _utils import read_defect_examples


class DefectDataTests(unittest.TestCase):
    def test_sample_limit_and_full_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            with path.open("w") as output:
                for index in range(3):
                    output.write(json.dumps({"code_change": str(index), "label": index % 2}) + "\n")
            self.assertEqual(len(read_defect_examples(path, 2)), 2)
            self.assertEqual(len(read_defect_examples(path)), 3)


if __name__ == "__main__":
    unittest.main()
