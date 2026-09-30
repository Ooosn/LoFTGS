import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


_INFERENCE_PATH = (
    Path(__file__).resolve().parents[1] / "gaussian_renderer" / "inference.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "loftgs_inference_under_test", _INFERENCE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load inference module from {_INFERENCE_PATH}")
inference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(inference)


class FakeASG(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.register_buffer("axis", torch.eye(3))
        self.property_hits = {"get_asg_lam_miu": 0, "get_asg_axis": 0}

    @property
    def get_asg_lam_miu(self):
        self.property_hits["get_asg_lam_miu"] += 1
        return self.scale.expand(1, 2, 1).clone()

    @property
    def get_asg_axis(self):
        self.property_hits["get_asg_axis"] += 1
        return self.axis.clone()

    def forward(self, value):
        return value * self.scale


class FakeModel:
    def __init__(self):
        self._scaling = torch.tensor([[1.0, 2.0, 3.0]])
        self._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self._opacity = torch.tensor([[0.25]])
        self._local_axis = torch.eye(3).unsqueeze(0)
        self._ks = torch.tensor([[0.1, 0.2, 0.3]])
        self._alpha_asg = torch.ones(1, 2)
        self._texture_color = torch.zeros(1, 3, 2, 2)
        self._texture_alpha = torch.ones(1, 1, 2, 2)
        self._texture_specular_gain = torch.full((1, 1, 2, 2), 0.5)
        self._texture_specular_lobe_scale = torch.full((1, 1, 2, 2), 0.75)
        self.get_xyz = torch.zeros(1, 3)

        self.asg_func = FakeASG()
        self.texture_effect_mode = "uvshadow_specular_lobe"
        self.texture_resolution = 2
        self.texture_dynamic_resolution = False
        self.texture_sigma_factor = 1.0
        self.mbrdf_normal_source = "local_q"
        self.active_sh_degree = 0
        self.frame_tag = "initial"
        self.property_hits = {name: 0 for name in inference._PROPERTIES}

    def _read(self, name, raw_name):
        self.property_hits[name] += 1
        return getattr(self, raw_name).clone()

    @property
    def get_scaling(self):
        return self._read("get_scaling", "_scaling")

    @property
    def get_rotation(self):
        return self._read("get_rotation", "_rotation")

    @property
    def get_opacity(self):
        return self._read("get_opacity", "_opacity")

    @property
    def get_local_axis(self):
        return self._read("get_local_axis", "_local_axis")

    @property
    def get_ks(self):
        return self._read("get_ks", "_ks")

    @property
    def get_alpha_asg(self):
        return self._read("get_alpha_asg", "_alpha_asg")

    @property
    def get_texture_color(self):
        return self._read("get_texture_color", "_texture_color")

    @property
    def get_texture_alpha(self):
        return self._read("get_texture_alpha", "_texture_alpha")

    @property
    def get_texture_specular_gain(self):
        return self._read("get_texture_specular_gain", "_texture_specular_gain")

    @property
    def get_texture_specular_lobe_scale(self):
        return self._read(
            "get_texture_specular_lobe_scale", "_texture_specular_lobe_scale"
        )


class TextureInferenceStateTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.state = inference.TextureInferenceState()

    def prepare(self):
        with torch.no_grad():
            return self.state.prepare(self.model)

    def test_prepared_view_requires_initialization(self):
        with self.assertRaisesRegex(RuntimeError, "must be prepared"):
            self.state.get_prepared()
        prepared = self.prepare()
        self.assertIs(self.state.get_prepared(), prepared)

    def test_cache_hits_and_non_cached_attributes_read_through(self):
        first = self.prepare()
        cached_values = {
            name: getattr(first, name).clone() for name in inference._PROPERTIES
        }

        self.model.frame_tag = "updated"
        second = self.prepare()

        self.assertIs(first, second)
        self.assertEqual(
            self.model.property_hits,
            {name: 1 for name in inference._PROPERTIES},
        )
        for name, expected in cached_values.items():
            self.assertTrue(torch.equal(getattr(second, name), expected), name)
        self.assertEqual(second.frame_tag, "updated")

        value = torch.tensor([3.0])
        self.assertTrue(torch.equal(second.asg_func(value), self.model.asg_func(value)))
        self.assertEqual(
            self.model.asg_func.property_hits,
            {"get_asg_lam_miu": 1, "get_asg_axis": 1},
        )

    def test_normal_in_place_tensor_mutation_rebuilds(self):
        first = self.prepare()
        with torch.no_grad():
            self.model._scaling.add_(1.0)
        second = self.prepare()

        self.assertIsNot(first, second)
        self.assertEqual(self.model.property_hits["get_scaling"], 2)
        self.assertFalse(torch.equal(first.get_scaling, second.get_scaling))

    def test_tensor_replacement_rebuilds(self):
        first = self.prepare()
        self.model._opacity = torch.full_like(self.model._opacity, 0.9)
        second = self.prepare()

        self.assertIsNot(first, second)
        self.assertEqual(self.model.property_hits["get_opacity"], 2)
        self.assertTrue(torch.equal(second.get_opacity, self.model._opacity))

    def test_dtype_and_cpu_conversion_rebuilds(self):
        first = self.prepare()
        self.model._ks = self.model._ks.to(dtype=torch.float64, device="cpu")
        second = self.prepare()

        self.assertIsNot(first, second)
        self.assertEqual(self.model.property_hits["get_ks"], 2)
        self.assertEqual(second.get_ks.dtype, torch.float64)
        self.assertEqual(second.get_ks.device.type, "cpu")

    def test_config_and_texture_layout_changes_rebuild(self):
        current = self.prepare()
        changes = (
            ("texture_effect_mode", "per_uv"),
            ("texture_resolution", 4),
            ("texture_dynamic_resolution", True),
            ("texture_sigma_factor", 1.5),
            ("mbrdf_normal_source", "2dgs"),
            ("active_sh_degree", 2),
        )
        for name, value in changes:
            with self.subTest(name=name):
                setattr(self.model, name, value)
                rebuilt = self.prepare()
                self.assertIsNot(current, rebuilt)
                current = rebuilt

        self.model._texture_color = torch.zeros(1, 3, 3, 3)
        rebuilt = self.prepare()
        self.assertIsNot(current, rebuilt)
        self.assertEqual(self.model.property_hits["get_texture_color"], 8)
        self.assertEqual(tuple(rebuilt.get_texture_color.shape), (1, 3, 3, 3))

    def test_asg_parameter_update_and_replacement_rebuild(self):
        first = self.prepare()
        first_lam = first.asg_func.get_asg_lam_miu.clone()
        with torch.no_grad():
            self.model.asg_func.scale.add_(1.0)
        second = self.prepare()

        self.assertIsNot(first, second)
        self.assertFalse(torch.equal(first_lam, second.asg_func.get_asg_lam_miu))

        self.model.asg_func.scale = nn.Parameter(torch.tensor(7.0))
        third = self.prepare()
        self.assertIsNot(second, third)
        self.assertTrue(torch.equal(third.asg_func.get_asg_lam_miu, torch.full((1, 2, 1), 7.0)))
        self.assertEqual(self.model.asg_func.property_hits["get_asg_lam_miu"], 3)
        self.assertEqual(self.model.asg_func.property_hits["get_asg_axis"], 3)

    def test_explicit_layout_version_rebuilds(self):
        first = self.prepare()
        self.model._dynamic_texture_layout_version = 1
        self.assertIsNot(first, self.prepare())

    def test_grad_enabled_prepare_is_rejected(self):
        with torch.enable_grad():
            with self.assertRaisesRegex(
                RuntimeError, "TextureInferenceState requires torch.no_grad"
            ):
                self.state.prepare(self.model)

        self.assertEqual(
            self.model.property_hits,
            {name: 0 for name in inference._PROPERTIES},
        )
        self.assertIsNotNone(self.prepare())

    def test_inference_mode_prepare_is_rejected(self):
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, "inference_mode"):
                self.state.prepare(self.model)

        self.assertEqual(
            self.model.property_hits,
            {name: 0 for name in inference._PROPERTIES},
        )
        self.assertIsNotNone(self.prepare())

    def test_clear_rebuilds_after_data_mutation(self):
        first = self.prepare()
        before = first.get_scaling.clone()
        self.model._scaling.data.add_(5.0)

        self.state.clear()
        rebuilt = self.prepare()

        self.assertIsNot(first, rebuilt)
        self.assertEqual(self.model.property_hits["get_scaling"], 2)
        self.assertFalse(torch.equal(before, rebuilt.get_scaling))
        self.assertTrue(torch.equal(rebuilt.get_scaling, self.model._scaling))


if __name__ == "__main__":
    unittest.main()
