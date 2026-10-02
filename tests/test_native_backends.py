"""CPU-only contracts for the packaged native 2DGS extension surface."""
import argparse
import ast
import importlib.util
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from arguments import ModelParams
from test_release_contracts import source_functions
from utils.native_backend import validate_native_backend


ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {
    "simple-knn": "simple_knn",
    "surfel-texture": "surfel_texture",
    "surfel-texture-deferred": "surfel_texture_deferred",
    "diff-surfel-rasterization-shadow": "diff_surfel_rasterization_shadow",
}
RETIRED_MODULES = {
    "diff_gaussian_rasterization", "diff_gaussian_rasterization_light",
    "diff_gaussian_rasterization_hgs", "v_3dgs", "v_3dgs_ortho", "gsplat", "gsplat_cuda",
}
RETIRED_PYTHON_MODULES = {"scene.gaussian_model", "gaussian_renderer.shadow_render"}
LIFTED_SHADOW_ENV = "GS3_2DGS_USE_LIFTED_SHADOW"


def load_benchmark():
    path = ROOT / "scripts/evaluation/benchmark_render.py"
    spec = importlib.util.spec_from_file_location("native_backend_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeExtensionManifestTests(unittest.TestCase):
    def test_requirements_install_exactly_four_native_extensions(self):
        lines = [line.partition("#")[0].strip()
                 for line in (ROOT / "requirements.txt").read_text().splitlines()]
        editable = [line for line in lines if line.startswith("-e ")]
        self.assertCountEqual(editable, [f"-e submodules/{name}" for name in EXTENSIONS])
        self.assertFalse(any(re.match(r"gsplat(?:$|[\s<>=!~@\[])", line) for line in lines))

    def test_only_four_native_build_targets_are_shipped(self):
        targets = {path.parent.name for path in (ROOT / "submodules").glob("*/setup.py")}
        self.assertEqual(targets, set(EXTENSIONS))
        self.assertTrue((ROOT / "submodules/third_party/glm/glm/glm.hpp").is_file())
        self.assertTrue((ROOT / "submodules/third_party/glm/copying.txt").is_file())

    def test_retired_extension_and_python_sources_are_absent(self):
        for folder in ("diff-gaussian-rasterization", "diff-gaussian-rasterization_light",
                       "diff-gaussian-rasterization_hgs", "v_3dgs", "v_3dgs_ortho"):
            with self.subTest(folder=folder):
                self.assertFalse((ROOT / "submodules" / folder).exists())
        for module in RETIRED_PYTHON_MODULES:
            with self.subTest(module=module):
                self.assertFalse((ROOT / (module.replace(".", "/") + ".py")).exists())

    def test_production_python_does_not_import_retired_modules(self):
        paths = list(ROOT.glob("*.py"))
        for folder in ("arguments", "gaussian_renderer", "scene", "utils", "scripts"):
            paths.extend((ROOT / folder).rglob("*.py"))
        violations = []
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level:
                        package = list(path.relative_to(ROOT).parent.parts)
                        if node.level > len(package):
                            continue
                        base = package[:len(package) - node.level + 1]
                        module = ".".join(base + ([module] if module else []))
                    names = [module] + [f"{module}.{alias.name}" for alias in node.names]
                elif isinstance(node, ast.Call) and node.args:
                    dynamic_import = (
                        isinstance(node.func, ast.Name) and node.func.id == "__import__"
                        or isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
                    )
                    if dynamic_import and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        names = [node.args[0].value]
                for name in names:
                    if name.partition(".")[0] in RETIRED_MODULES or any(
                            name == old or name.startswith(old + ".") for old in RETIRED_PYTHON_MODULES):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
        self.assertEqual(violations, [])


class NativeBackendValidationTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {LIFTED_SHADOW_ENV: "0"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_missing_backend_and_cli_default_select_native_2dgs(self):
        self.assertEqual(validate_native_backend(SimpleNamespace()), "2dgs")
        parser = argparse.ArgumentParser()
        ModelParams(parser)
        args = parser.parse_args([])
        self.assertEqual(args.rasterizer, "2dgs")
        self.assertFalse(args.use_hgs)
        self.assertFalse(args.use_hgs_finetune)
        self.assertEqual(validate_native_backend(args), "2dgs")

    def test_both_native_backends_are_accepted(self):
        for backend in ("2dgs", "2dgs_3ch"):
            with self.subTest(backend=backend):
                self.assertEqual(validate_native_backend(SimpleNamespace(rasterizer=backend)), backend)

    def test_retired_unknown_and_null_backends_are_rejected(self):
        for backend in ("3dgs", "gsplat", "hgs", "unknown", "", None):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "Unsupported rasterizer"):
                validate_native_backend(SimpleNamespace(rasterizer=backend))

    def test_hgs_flags_are_rejected_independently(self):
        for flag in ("use_hgs", "use_hgs_finetune"):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "HGS"):
                validate_native_backend(SimpleNamespace(rasterizer="2dgs", **{flag: True}))

    def test_lifted_shadow_opt_in_is_rejected(self):
        with patch.dict(os.environ, {LIFTED_SHADOW_ENV: "1"}), \
                self.assertRaisesRegex(ValueError, LIFTED_SHADOW_ENV):
            validate_native_backend(SimpleNamespace(rasterizer="2dgs"))

    def test_cli_legacy_choices_fail_validation(self):
        parser = argparse.ArgumentParser()
        ModelParams(parser)
        for flags in (["--rasterizer", "gsplat"], ["--use_hgs"], ["--use_hgs_finetune"]):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                validate_native_backend(parser.parse_args(flags))
        self.assertEqual(validate_native_backend(parser.parse_args(["--rasterizer", "2dgs_3ch"])), "2dgs_3ch")

    def adapter_factory(self):
        native = Mock(return_value=object())
        namespace = {"_NativeTextureAdapter": native, "validate_native_backend": validate_native_backend}
        source_functions("scene/gaussian_model_2dgs_adapter.py", ["__new__"], namespace,
                         class_name="GaussianModel2DGSAdapter")
        return namespace["__new__"], native

    def test_adapter_preserves_textured_and_untextured_native_construction(self):
        for backend in ("2dgs", "2dgs_3ch"):
            for textured in (False, True):
                with self.subTest(backend=backend, textured=textured):
                    factory, native = self.adapter_factory()
                    model = SimpleNamespace(rasterizer=backend, use_textures=textured)
                    opt = object()
                    self.assertIs(factory(None, model, opt), native.return_value)
                    native.assert_called_once_with(model, opt)

    def test_adapter_rejects_invalid_configuration_before_native_construction(self):
        configurations = ({"rasterizer": "gsplat"}, {"use_hgs": True}, {"use_hgs_finetune": True})
        for values in configurations:
            with self.subTest(values=values):
                factory, native = self.adapter_factory()
                with self.assertRaises(ValueError):
                    factory(None, SimpleNamespace(**values))
                native.assert_not_called()
        factory, native = self.adapter_factory()
        with patch.dict(os.environ, {LIFTED_SHADOW_ENV: "1"}), self.assertRaises(ValueError):
            factory(None, SimpleNamespace())
        native.assert_not_called()


class NativeRendererDispatchTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {LIFTED_SHADOW_ENV: "0"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.calls = {name: Mock(return_value=object()) for name in (
            "_render_2dgs_native_3ch", "_render_2dgs_native_deferred", "render_2dgs_texture_deferred")}
        self.validator = Mock(wraps=validate_native_backend)
        namespace = {"GaussianModel": object, "torch": SimpleNamespace(Tensor=object),
                     "validate_native_backend": self.validator, **self.calls}
        source_functions("gaussian_renderer/__init__.py", ["render"], namespace)
        self.render = namespace["render"]
        self.values = {name: object() for name in (
            "viewpoint_camera", "light_stream", "calc_stream", "local_axises", "asg_scales",
            "asg_axises", "pipe", "bg_color", "override_color", "inference_state")}
        self.model = SimpleNamespace(rasterizer="2dgs")

    def invoke(self, gau, model=None, **overrides):
        values = {**self.values, "gau": gau, "modelset": self.model if model is None else model,
                  "scaling_modifier": 1.75, "fix_labert": True, "iteration": 321,
                  "inten_scale": 2.5, "is_train": True, "asg_mlp": True, **overrides}
        return self.render(**values)

    def gaussian(self, backend="2dgs", textured=False):
        return SimpleNamespace(rasterizer=backend, use_textures=textured,
                               get_scaling=SimpleNamespace(shape=(7, 2)))

    def assert_branch(self, gau, selected, textured=False):
        result = self.invoke(gau)
        kwargs = {"scaling_modifier": 1.75, "fix_labert": True, "iteration": 321}
        if textured:
            kwargs.update({name: self.values[name] for name in ("light_stream", "calc_stream", "inference_state")})
        self.calls[selected].assert_called_once_with(
            self.values["viewpoint_camera"], gau, self.values["pipe"], self.values["bg_color"], self.model, **kwargs)
        self.assertIs(result, self.calls[selected].return_value)
        self.assertEqual([c.args[0] for c in self.validator.call_args_list], [self.model, gau])
        for name, function in self.calls.items():
            if name != selected:
                function.assert_not_called()

    def test_textured_dispatch_forwards_original_native_arguments(self):
        self.assert_branch(self.gaussian(textured=True), "render_2dgs_texture_deferred", textured=True)

    def test_untextured_initialization_uses_native_deferred(self):
        self.assert_branch(self.gaussian(), "_render_2dgs_native_deferred")

    def test_three_channel_backend_has_priority_over_texture_flag(self):
        self.assert_branch(self.gaussian("2dgs_3ch", textured=True), "_render_2dgs_native_3ch")

    def test_three_channel_untextured_backend_is_preserved(self):
        self.assert_branch(self.gaussian("2dgs_3ch"), "_render_2dgs_native_3ch")

    def assert_no_native_calls(self):
        for function in self.calls.values():
            function.assert_not_called()

    def forbidden_geometry(self, backend="2dgs", **flags):
        class NoGeometry:
            @property
            def get_scaling(self):
                raise AssertionError("Geometry was accessed before configuration rejection")
        gau = NoGeometry()
        gau.rasterizer = backend
        for key, value in flags.items():
            setattr(gau, key, value)
        return gau

    def test_model_and_gaussian_backends_are_both_validated_before_geometry(self):
        cases = ((SimpleNamespace(rasterizer="gsplat"), self.forbidden_geometry()),
                 (self.model, self.forbidden_geometry("gsplat")))
        for model, gau in cases:
            with self.subTest(model=model.rasterizer, gaussian=gau.rasterizer), \
                    self.assertRaisesRegex(ValueError, "Unsupported rasterizer"):
                self.invoke(gau, model=model)
        self.assert_no_native_calls()

    def test_model_and_gaussian_hgs_flags_fail_before_geometry(self):
        for flag in ("use_hgs", "use_hgs_finetune"):
            for owner in ("model", "gaussian"):
                model = SimpleNamespace(rasterizer="2dgs", **({flag: True} if owner == "model" else {}))
                gau = self.forbidden_geometry(**({flag: True} if owner == "gaussian" else {}))
                with self.subTest(flag=flag, owner=owner), self.assertRaisesRegex(ValueError, "HGS"):
                    self.invoke(gau, model=model)
        self.assert_no_native_calls()

    def test_lifted_shadow_and_shadowmap_render_fail_before_geometry(self):
        with patch.dict(os.environ, {LIFTED_SHADOW_ENV: "1"}), self.assertRaisesRegex(ValueError, LIFTED_SHADOW_ENV):
            self.invoke(self.forbidden_geometry())
        with self.assertRaisesRegex(ValueError, "shadowmap_render"):
            self.invoke(self.forbidden_geometry(), shadowmap_render=True)
        self.assert_no_native_calls()

    def test_non_surfel_geometry_is_rejected_before_dispatch(self):
        for axes in (1, 3):
            gau = self.gaussian()
            gau.get_scaling.shape = (7, axes)
            with self.subTest(axes=axes), self.assertRaisesRegex(ValueError, "two-axis"):
                self.invoke(gau)
        with self.assertRaisesRegex(ValueError, "two-axis"):
            self.invoke(SimpleNamespace(rasterizer="2dgs"))
        self.assert_no_native_calls()


class NativeBenchmarkImportTests(unittest.TestCase):
    def setUp(self):
        self.benchmark = load_benchmark()

    def test_package_manifest_and_provenance_prefixes_exclude_retired_modules(self):
        self.assertEqual(self.benchmark.EXTENSION_PACKAGES, EXTENSIONS)
        self.assertCountEqual(self.benchmark.EXTENSION_NAMES, EXTENSIONS.values())
        prefixes = self.benchmark.CUDA_MODULE_PREFIXES
        self.assertTrue(all(package.startswith(prefixes) for package in EXTENSIONS.values()))
        self.assertTrue("tinycudann".startswith(prefixes))
        for name in RETIRED_MODULES | {"gsplat_cuda"}:
            self.assertFalse(name.startswith(prefixes), name)

    def test_configure_imports_uses_hyphenated_source_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            native_root = Path(temporary)
            for folder in list(EXTENSIONS) + list(EXTENSIONS.values()) + ["v_3dgs"]:
                (native_root / "submodules" / folder).mkdir(parents=True, exist_ok=True)
            with patch.object(sys, "path", ["unrelated-path"]), \
                    patch.dict(os.environ, {"GS3_ALLOW_TORCH_PHASE_FALLBACK": "1"}):
                self.benchmark._configure_imports(native_root)
                expected = {str(native_root / "submodules" / name) for name in EXTENSIONS}
                self.assertTrue(expected.issubset(sys.path))
                self.assertEqual(sys.path[0], str(self.benchmark.REPO_ROOT))
                for package in list(EXTENSIONS.values()) + ["v_3dgs"]:
                    self.assertNotIn(str(native_root / "submodules" / package), sys.path)
                first_paths = list(sys.path)
                self.benchmark._configure_imports(native_root)
                self.assertEqual(sys.path, first_paths)
                self.assertNotIn("GS3_ALLOW_TORCH_PHASE_FALLBACK", os.environ)
                self.assertEqual(os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"], "1")
                self.assertEqual(os.environ["OPENCV_IO_ENABLE_OPENEXR"], "1")

    def test_missing_native_root_fails_without_changing_import_paths(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(sys, "path", ["unchanged"]):
            with self.assertRaisesRegex(RuntimeError, "Native extension root is missing"):
                self.benchmark._configure_imports(Path(temporary) / "missing")
            self.assertEqual(sys.path, ["unchanged"])


if __name__ == "__main__":
    unittest.main()
