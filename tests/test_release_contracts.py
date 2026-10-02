"""CPU regressions for CLI and render orchestration; rasterization is mocked."""
import argparse
import ast
import contextlib
import io
import os
from pathlib import Path
import shlex
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from arguments import ModelParams, OptimizationParams, PipelineParams, explicit_cli_value
from utils.shadow_transport import checkpoint_shadow_anchor, shadow_sensitivity_anchor
from utils.native_backend import validate_native_backend


ROOT = Path(__file__).resolve().parents[1]


def source_functions(relative_path, names, namespace, class_name=None):
    # Import orchestration without importing the package's CUDA extensions.
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    if class_name:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == class_name)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError(f"Missing tested functions in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ArgumentContractTests(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        ModelParams(parser)
        OptimizationParams(parser)
        PipelineParams(parser)
        return parser

    def test_default_and_both_anchor_values(self):
        parser = self.parser()
        self.assertEqual(parser.parse_args([]).texture_shadow_sensitivity_anchor, "no_shadow")
        for anchor in ("no_shadow", "visibility"):
            args = parser.parse_args(["--texture_shadow_sensitivity_anchor", anchor])
            self.assertEqual(args.texture_shadow_sensitivity_anchor, anchor)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--texture_shadow_sensitivity_anchor", "misspelled"])

    def test_explicit_override_is_distinct_from_a_default_or_saved_value(self):
        args = SimpleNamespace(texture_shadow_sensitivity_anchor="visibility")
        name = "texture_shadow_sensitivity_anchor"
        self.assertIsNone(explicit_cli_value(args, name, []))
        self.assertEqual(explicit_cli_value(args, name, ["--" + name, "visibility"]), "visibility")
        args.texture_shadow_sensitivity_anchor = "no_shadow"
        self.assertEqual(explicit_cli_value(args, name, ["--" + name + "=no_shadow"]), "no_shadow")
        self.assertEqual(explicit_cli_value(args, name, ["--" + name + "=visibility",
                                                        "--" + name, "no_shadow"]), "no_shadow")

    def test_checkpoint_anchor_metadata(self):
        self.assertIsNone(checkpoint_shadow_anchor(None))
        self.assertIsNone(checkpoint_shadow_anchor(tuple(range(16))))
        for anchor in ("no_shadow", "visibility"):
            params = [None] * 32 + [{"texture_shadow_sensitivity_anchor": anchor}]
            self.assertEqual(checkpoint_shadow_anchor(params), anchor)
        with self.assertRaises(ValueError):
            checkpoint_shadow_anchor([None] * 32 + [{"texture_shadow_sensitivity_anchor": "bad"}])

    def test_training_resolves_anchor_before_writing_run_configuration(self):
        class StopBeforeGPU(Exception):
            pass

        for saved, override, expected in ((None, None, "no_shadow"),
                                          ("visibility", None, "visibility"),
                                          ("visibility", "no_shadow", "no_shadow"),
                                          ("no_shadow", "visibility", "visibility")):
            model = SimpleNamespace(source_path="dataset", rasterizer="2dgs",
                                    texture_shadow_sensitivity_anchor="no_shadow")
            prepare = Mock()
            namespace = {"torch": torch, "checkpoint_shadow_anchor": checkpoint_shadow_anchor,
                         "shadow_sensitivity_anchor": shadow_sensitivity_anchor,
                         "_assert_checkpoint_source_matches": Mock(),
                         "prepare_output_and_logger": prepare,
                         "GaussianModel2DGSAdapter": Mock(side_effect=StopBeforeGPU),
                         "print": lambda *args: None}
            source_functions("train.py", ["training"], namespace)
            metadata = {} if saved is None else {"texture_shadow_sensitivity_anchor": saved}
            with patch("torch.load", return_value=([None] * 32 + [metadata], 1000)), \
                    self.assertRaises(StopBeforeGPU):
                namespace["training"](model, SimpleNamespace(iterations=100000), None,
                                      [], [], [], "checkpoint.pth", 0, -1,
                                      shadow_anchor_override=override)
            self.assertEqual(prepare.call_args.args[0].texture_shadow_sensitivity_anchor, expected)

    def test_shipped_training_commands_keep_their_numeric_settings(self):
        parser = self.parser()
        for name in ("test_iterations", "save_iterations", "checkpoint_iterations"):
            parser.add_argument("--" + name, nargs="+", type=int)
        parser.add_argument("--unfreeze_iterations", type=int)
        script = (ROOT / "train.sh").read_text().replace("\\\n", "")
        commands = [line for line in script.splitlines() if line.startswith("python train.py")]
        self.assertEqual(len(commands), 2)
        for command, densify in zip(commands, (50000, 80000)):
            args = parser.parse_args(shlex.split(command)[2:])
            self.assertEqual(args.iterations, 100000)
            self.assertEqual(args.densify_until_iter, densify)
            self.assertEqual(args.texture_resolution, 4)
            self.assertEqual(args.texture_shadow_sensitivity_anchor, "no_shadow")


class CPUCamera:
    def __init__(self):
        self.image_name = "frame"
        self.R_cu = torch.eye(3)
        self.T_cu = torch.zeros(3)
        self.cam_pose_adj = torch.zeros(1, 6)
        self.pl_adj = torch.zeros(1, 3)
        self.pl_pos_init = torch.tensor([[3.0, 0.0, 0.0]])
        self.update()

    def update(self, mode="SO3xR3"):
        self.pl_pos = self.pl_pos_init + self.pl_adj


SCENE_METHODS = source_functions(
    "scene/__init__.py", ["_apply_optimized_camera_frame", "restore"],
    {"torch": torch, "np": np}, class_name="Scene")


class CPUScene:
    _apply_optimized_camera_frame = SCENE_METHODS["_apply_optimized_camera_frame"]
    restore = SCENE_METHODS["restore"]

    def __init__(self, model, gaussians, **kwargs):
        self.load_optimized_cameras = kwargs["load_optimized_cameras"]
        self.loaded_iter = kwargs["load_iteration"]
        self.source_path = model.source_path
        self.camera = CPUCamera()
        self.train_cameras = {1.0: []}
        self.test_cameras = {1.0: [self.camera]}
        self.optimizer = None
        if self.load_optimized_cameras or "iteration_" in self.source_path:
            self._apply_optimized_camera_frame(self.camera, {
                "R_opt": np.eye(3).tolist(), "T_opt": [0, 0, 0], "pl_pos": [3.1, 0, 0]})

    def getTestCameras(self):
        return [self.camera]


class RenderContractTests(unittest.TestCase):
    def run_render(self, *, opt_pose=False, force_input=False, scene_state=True,
                   saved_anchor="visibility", override=None, missing_checkpoint=False):
        residuals = {"cameras": {"test": {"1.0": [{"image_name": "frame",
                     "cam_pose_adj": torch.zeros(1, 6), "pl_adj": torch.tensor([[0.1, 0, 0]])}]}}}
        model = SimpleNamespace(model_path="model", source_path="dataset", rasterizer="2dgs",
                                use_nerual_phasefunc=True, use_textures=False, white_background=False)
        gaussians = SimpleNamespace(neural_phasefunc=Mock(), training_setup=Mock(), optimizer=None)

        def restore(*args, **kwargs):
            gaussians.texture_shadow_sensitivity_anchor = saved_anchor

        gaussians.restore = restore
        scenes = []

        def scene_factory(*args, **kwargs):
            scene = CPUScene(*args, **kwargs)
            scenes.append(scene)
            return scene

        namespace = {"torch": torch, "os": os, "ModelParams": ModelParams,
                     "PipelineParams": PipelineParams, "Scene": scene_factory,
                     "_make_gaussian_model": lambda *args: gaussians,
                     "searchForMaxIteration": Mock(return_value=100000),
                     "shadow_sensitivity_anchor": shadow_sensitivity_anchor,
                     "render_set": Mock(), "print": lambda *args: None}
        source_functions("render.py", ["render_sets", "_unpack_training_checkpoint", "_is_2dgs_model"], namespace)
        real_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs.pop("device", None)
            return real_tensor(*args, **kwargs)

        with patch.dict(os.environ, {"MICRO_FORCE_INPUT_CAMERA": "1" if force_input else "0"}), \
                patch("os.path.isfile", return_value=not missing_checkpoint), \
                patch("torch.load", return_value=([], 100000, residuals if scene_state else None)) as load, \
                patch("torch.tensor", side_effect=cpu_tensor), patch("torch.optim.Adam"), \
                patch("torch.cuda.empty_cache"):
            namespace["render_sets"](model, -1, SimpleNamespace(), True, False, opt_pose,
                                      False, False, False, False, False, False, False,
                                      opt=SimpleNamespace(), shadow_anchor_override=override)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(Path(load.call_args.args[0]).name, "chkpnt100000.pth")
        return scenes[0], gaussians

    def test_checkpoint_residuals_apply_once_without_opt_pose(self):
        scene, _ = self.run_render()
        self.assertFalse(scene.load_optimized_cameras)
        self.assertAlmostEqual(scene.camera.pl_pos[0, 0].item(), 3.1, places=5)

    def test_legacy_checkpoint_without_scene_state_uses_baked_transforms(self):
        scene, _ = self.run_render(scene_state=False)
        self.assertTrue(scene.load_optimized_cameras)
        self.assertAlmostEqual(scene.camera.pl_pos[0, 0].item(), 3.1, places=5)

    def test_opt_pose_resolves_latest_iteration_before_selecting_transforms(self):
        scene, _ = self.run_render(opt_pose=True)
        self.assertEqual(Path(scene.source_path).name, "iteration_100000")
        self.assertFalse(scene.load_optimized_cameras)
        self.assertAlmostEqual(scene.camera.pl_pos[0, 0].item(), 3.1, places=5)

    def test_force_input_camera_applies_neither_saved_representation(self):
        scene, _ = self.run_render(force_input=True)
        self.assertFalse(scene.load_optimized_cameras)
        self.assertAlmostEqual(scene.camera.pl_pos[0, 0].item(), 3.0, places=5)

    def test_render_preserves_saved_anchor_unless_explicitly_overridden(self):
        for saved in ("no_shadow", "visibility"):
            _, model = self.run_render(saved_anchor=saved)
            self.assertEqual(model.texture_shadow_sensitivity_anchor, saved)
            for override in ("no_shadow", "visibility"):
                _, model = self.run_render(saved_anchor=saved, override=override)
                self.assertEqual(model.texture_shadow_sensitivity_anchor, override)

    def test_missing_checkpoint_fails_before_scene_loading(self):
        with self.assertRaises(FileNotFoundError):
            self.run_render(missing_checkpoint=True)

    def test_no_texture_dispatch_has_no_legacy_directory_dependency(self):
        namespace = {"_NativeTextureAdapter": lambda *args: "native",
                     "validate_native_backend": validate_native_backend}
        source_functions("scene/gaussian_model_2dgs_adapter.py", ["__new__"], namespace,
                         class_name="GaussianModel2DGSAdapter")
        for textures in (False, True):
            self.assertEqual(namespace["__new__"](None, SimpleNamespace(use_textures=textures)), "native")


if __name__ == "__main__":
    unittest.main()
