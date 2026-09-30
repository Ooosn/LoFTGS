import argparse
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arguments import ModelParams, PipelineParams, get_combined_args, read_model_config
from utils.system_utils import searchForMaxIteration
from test_release_contracts import source_functions


ROOT = Path(__file__).resolve().parents[1]


class ModelConfigTests(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        ModelParams(parser, sentinel=True)
        PipelineParams(parser)
        return parser

    def test_saved_flags_are_not_silently_reset_and_explicit_cli_wins(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp) / "cfg_args"
            path.write_text("Namespace(source_path='data', hdr=True, detach_shadow=True, "
                            "texture_shadow_confidence_enabled=True, "
                            "texture_shadow_sensitivity_anchor='visibility')", encoding="utf-8")
            with patch("sys.argv", ["render.py", "-m", temp]):
                args = get_combined_args(self.parser())
            self.assertTrue(args.hdr)
            self.assertTrue(args.detach_shadow)
            self.assertTrue(args.texture_shadow_confidence_enabled)
            self.assertEqual(args.texture_shadow_sensitivity_anchor, "visibility")
            with patch("sys.argv", ["render.py", "-m", temp, "--no_detach_shadow",
                                    "--texture_shadow_sensitivity_anchor=no_shadow"]):
                args = get_combined_args(self.parser())
            self.assertFalse(args.detach_shadow)
            self.assertEqual(args.texture_shadow_sensitivity_anchor, "no_shadow")

    def test_missing_or_malformed_config_is_not_replaced_by_defaults(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            with patch("sys.argv", ["render.py", "-m", temp]), self.assertRaises(FileNotFoundError):
                get_combined_args(self.parser())
            path = Path(temp) / "cfg_args"
            path.write_text("Namespace(hdr=", encoding="utf-8")
            with patch("sys.argv", ["render.py", "-m", temp]), self.assertRaises(ValueError):
                get_combined_args(self.parser())

    def test_configuration_does_not_execute_expressions(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp) / "cfg_args"
            path.write_text("Namespace(hdr=__import__('os').getcwd())", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid model configuration"):
                read_model_config(path)

    def test_literal_paths_and_containers_round_trip(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp) / "cfg_args"
            original = argparse.Namespace(source_path="D:/data/it's a scene", hdr=False,
                                          test_iterations=[1000, 100000], note=None)
            path.write_text(str(original), encoding="utf-8")
            self.assertEqual(vars(read_model_config(path)), vars(original))

    def test_training_resume_source_paths_support_apostrophes(self):
        namespace = {"Path": Path, "os": os, "read_model_config": read_model_config}
        source_functions("train.py", ["_normalize_compare_path", "_checkpoint_cfg_source_path"], namespace)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            source = str(Path(temp) / "a scene's data")
            (Path(temp) / "cfg_args").write_text(str(argparse.Namespace(source_path=source)), encoding="utf-8")
            actual = namespace["_checkpoint_cfg_source_path"](Path(temp) / "chkpnt100000.pth")
            self.assertEqual(actual, os.path.realpath(os.path.abspath(source)))

    def test_training_only_dataset_read_initializes_the_test_split(self):
        class StopAfterNormalization(Exception):
            pass

        camera = object()
        def normalization(cameras):
            self.assertEqual(cameras, [camera])
            raise StopAfterNormalization

        namespace = {"print": lambda *args: None,
                     "readCamerasFromTransforms": lambda *args: [camera],
                     "getNerfppNorm": normalization}
        source_functions("scene/dataset_readers.py", ["readNerfSyntheticInfo"], namespace)
        with self.assertRaises(StopAfterNormalization):
            namespace["readNerfSyntheticInfo"]("data", False, False, 2000, 400, skip_test=True)

    def test_latest_iteration_ignores_unrelated_entries(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            for name in ("iteration_7000", "iteration_100000", "backup"):
                (root / name).mkdir()
            (root / "README.txt").touch()
            (root / "iteration_999999").touch()
            self.assertEqual(searchForMaxIteration(root), 100000)

    def test_latest_iteration_reports_an_empty_model_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, self.assertRaises(FileNotFoundError):
            searchForMaxIteration(temp)


if __name__ == "__main__":
    unittest.main()
