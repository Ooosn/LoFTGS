import contextlib
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "image_metrics_cli", ROOT / "scripts/evaluation/unified_image_metrics.py")
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


class StubLPIPS(torch.nn.Module):
    def forward(self, image, target):
        return (image - target).abs().mean(dim=(1, 2, 3), keepdim=True)


class ImageMetricsCLITests(unittest.TestCase):
    def test_mixed_resolutions_are_rejected_across_batch_boundaries(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            renders, gt = root / "renders", root / "gt"
            renders.mkdir()
            gt.mkdir()
            for index, size in enumerate(((12, 12), (16, 16))):
                Image.new("RGB", size, (10, 10, 10)).save(renders / f"{index}.png")
                Image.new("RGB", size).save(gt / f"{index}.png")
            with self.assertRaisesRegex(ValueError, "mixed resolutions"):
                metrics.evaluate(renders, gt, torch.device("cpu"), StubLPIPS(), 1)

    def test_nonfinite_lpips_is_rejected(self):
        class NonfiniteLPIPS(StubLPIPS):
            def forward(self, image, target):
                return super().forward(image, target) * float("nan")

        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            renders, gt = root / "renders", root / "gt"
            renders.mkdir()
            gt.mkdir()
            Image.new("RGB", (12, 12), (10, 10, 10)).save(renders / "0.png")
            Image.new("RGB", (12, 12)).save(gt / "0.png")
            with self.assertRaises(FloatingPointError):
                metrics.evaluate(renders, gt, torch.device("cpu"), NonfiniteLPIPS(), 1)

    def test_missing_images_fail_before_lpips_initialization(self):
        stub = SimpleNamespace(LPIPS=lambda net: self.fail("LPIPS should not initialize"))
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, patch.dict("sys.modules", {"lpips": stub}):
            with self.assertRaisesRegex(ValueError, "no rendered images"):
                metrics.main(["--renders", str(Path(temp) / "missing"), "--gt", temp,
                              "--out", str(Path(temp) / "metrics")])
            self.assertFalse((Path(temp) / "metrics").exists())

    def test_direct_directories_need_no_manifest(self):
        args, entries = metrics.parse_inputs([
            "--renders", "rendered images", "--gt", "ground truth", "--out", "metrics",
            "--scene", "Hotdog"])
        self.assertIsNone(args.manifest)
        self.assertEqual(entries, [{"scene": "Hotdog", "method": "LoFT-GS",
                                    "renders": "rendered images", "gt": "ground truth"}])

    def test_invalid_cli_combinations_are_rejected(self):
        for argv in (["--renders", "pred", "--out", "out"],
                     ["--renders", "pred", "--manifest", "list.jsonl", "--out", "out"],
                     ["--manifest", "list.jsonl", "--gt", "gt", "--out", "out"],
                     ["--renders", "pred", "--gt", "gt", "--out", "out", "--batch-size", "0"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                metrics.parse_inputs(argv)

    def test_existing_batch_manifest_remains_compatible(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp) / "inputs.jsonl"
            entry = {"scene": "Hotdog", "method": "Ours", "renders": "pred", "gt": "gt"}
            path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
            _, entries = metrics.parse_inputs(["--manifest", str(path), "--out", "metrics"])
            self.assertEqual(entries, [entry])

    def test_direct_cli_writes_metrics_computed_from_image_pixels(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            renders, gt, out = root / "rendered images", root / "ground truth", root / "metrics"
            renders.mkdir()
            gt.mkdir()
            for index, error in enumerate((10, 20)):
                Image.new("RGB", (12, 12), (error,) * 3).save(renders / f"{index:05d}.png")
                Image.new("RGB", (12, 12)).save(gt / f"{index:05d}.png")
            stub = SimpleNamespace(LPIPS=lambda net: StubLPIPS())
            with patch.dict("sys.modules", {"lpips": stub}), contextlib.redirect_stdout(io.StringIO()):
                result = metrics.main(["--renders", str(renders), "--gt", str(gt),
                                       "--scene", "Hotdog", "--out", str(out), "--device", "cpu"])
            self.assertEqual(result, 0)
            with (out / "summary.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["count"], "2")
            self.assertEqual(row["scene"], "Hotdog")
            expected_psnr = sum(20 * math.log10(255 / value) for value in (10, 20)) / 2
            self.assertAlmostEqual(float(row["psnr"]), expected_psnr, places=5)
            self.assertTrue((out / "summary.md").is_file())
            with (out / "per_frame/Hotdog/LoFT-GS.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)


if __name__ == "__main__":
    unittest.main()
