import ast
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "scripts/evaluation/benchmark_render.py"
SPEC = importlib.util.spec_from_file_location("benchmark_render_cli", SOURCE)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkCLITests(unittest.TestCase):
    def test_minimal_command_does_not_need_render_control_flags(self):
        parser, _, _, _ = benchmark._parser()
        args = parser.parse_args(["-m", "output/Hotdog", "--views", "32", "--repeats", "3"])
        self.assertEqual(args.views, 32)
        self.assertEqual(args.repeats, 3)
        self.assertIsNone(args.memory_fraction)
        for name in ("skip_train", "skip_test", "opt_pose", "gamma", "valid", "write_images",
                     "force_save", "synthesize_video", "shadowmap_render"):
            self.assertNotIn("--" + name, parser._option_string_actions)

    def test_missing_tcnn_is_rejected_even_when_training_fallback_is_enabled(self):
        for enabled in ("0", "1"):
            with patch.dict(os.environ, {"GS3_ALLOW_TORCH_PHASE_FALLBACK": enabled}), \
                    patch.object(benchmark.importlib, "import_module", side_effect=ImportError("missing")), \
                    self.assertRaisesRegex(ImportError, "Render timing requires tinycudann"):
                benchmark._require_tcnn()

    def test_tcnn_requirement_checks_the_actual_package(self):
        with patch.object(benchmark.importlib, "import_module") as load:
            benchmark._require_tcnn()
        load.assert_called_once_with("tinycudann")

    def test_benchmark_keeps_fixed_camera_and_output_policy(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "render_sets"]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        expected = {"skip_train": True, "skip_test": False, "opt_pose": True,
                    "gamma": False, "valid": False, "write_images": False,
                    "force_save": False, "synthesize_video": False, "shadowmap_render": False}
        for name, value in expected.items():
            self.assertEqual(ast.literal_eval(keywords[name]), value)


if __name__ == "__main__":
    unittest.main()
