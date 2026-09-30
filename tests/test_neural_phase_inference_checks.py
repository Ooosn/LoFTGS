import builtins
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

_PHASE_SOURCE = Path(__file__).resolve().parents[1] / "scene" / "neural_phase_function.py"


def load_phase_without_tcnn(allow_fallback):
    module_name = f"neural_phase_without_tcnn_{int(allow_fallback)}"
    spec = importlib.util.spec_from_file_location(module_name, _PHASE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def import_without_tcnn(name, *args, **kwargs):
        if name == "tinycudann":
            raise ImportError("simulated missing tinycudann")
        return original_import(name, *args, **kwargs)

    with patch.dict(os.environ, {"GS3_ALLOW_TORCH_PHASE_FALLBACK": "1" if allow_fallback else "0"}):
        with patch("builtins.__import__", side_effect=import_without_tcnn):
            spec.loader.exec_module(module)
    return module


Neural_phase = load_phase_without_tcnn(False).Neural_phase


class IdentityEncoding(nn.Module):
    def forward(self, value):
        return value


class ConstantNetwork(nn.Module):
    def __init__(self, width, value=0.25):
        super().__init__()
        self.width = width
        self.value = value

    def forward(self, value):
        fill_value = 0.0 if self.value == "nan" else self.value
        result = value.new_full((value.shape[0], self.width), fill_value)
        if self.value == "nan":
            result.fill_(float("nan"))
        return result


class NeuralPhaseInferenceCheckTests(unittest.TestCase):
    def test_missing_tcnn_fails_by_default(self):
        module = load_phase_without_tcnn(False)
        with self.assertRaisesRegex(ImportError, "tinycudann is required"):
            module.Neural_phase()

    def test_torch_fallback_requires_explicit_opt_in(self):
        with self.assertWarnsRegex(RuntimeWarning, "GS3_ALLOW_TORCH_PHASE_FALLBACK=1"):
            module = load_phase_without_tcnn(True)
        phase = module.Neural_phase(hidden_feature_size=8, hidden_feature_layers=1, frequency=1)
        self.assertIsInstance(phase.shadow_func, module._TinyLikeNetwork)

    def test_explicit_torch_backend_supports_training_backward(self):
        with self.assertWarns(RuntimeWarning):
            module = load_phase_without_tcnn(True)
        phase = module.Neural_phase(hidden_feature_size=8, hidden_feature_layers=1, frequency=1)
        wi = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        wo = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        shadow, other, _, _ = phase(wi, wo, torch.zeros(2, 3), torch.ones(2, 6),
                                     hint=torch.full((2, 1), 0.5))
        (shadow.mean() + other.mean()).backward()
        grads = [parameter.grad for parameter in phase.parameters() if parameter.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))
        self.assertTrue(any(bool(grad.abs().sum() > 0) for grad in grads))

    def make_model(self):
        model = Neural_phase.__new__(Neural_phase)
        nn.Module.__init__(model)
        model.encoding = IdentityEncoding()
        model.other_effects_func = ConstantNetwork(3)
        model.shadow_func = ConstantNetwork(1)
        model.shadow_view_independent = False
        return model

    def inputs(self, bad=False):
        wi = torch.tensor([[1.0, 0.0, 0.0]])
        wo = torch.tensor([[0.0, 1.0, 0.0]])
        pos = torch.tensor([[0.0, 0.0, 1.0]])
        material = torch.tensor([[float("nan") if bad else 0.5]])
        hint = torch.tensor([[0.8]])
        return wi, wo, pos, material, hint

    def test_training_default_keeps_finite_input_and_output_checks(self):
        model = self.make_model()
        wi, wo, pos, material, hint = self.inputs(bad=True)
        with self.assertRaisesRegex(FloatingPointError, "Non-finite input"):
            model(wi, wo, pos, material, hint=hint)

        model.other_effects_func.value = "nan"
        wi, wo, pos, material, hint = self.inputs()
        with self.assertRaisesRegex(FloatingPointError, "produced non-finite outputs"):
            model(wi, wo, pos, material, hint=hint)

    def test_inference_gate_skips_only_the_synchronizing_scans(self):
        model = self.make_model()
        wi, wo, pos, material, hint = self.inputs()
        with patch("torch.isfinite", wraps=torch.isfinite) as finite:
            shadow, other, _, _ = model(wi, wo, pos, material, hint=hint, validate_finite=False)
        self.assertEqual(finite.call_count, 0)
        self.assertTrue(torch.isfinite(shadow).all())
        self.assertTrue(torch.isfinite(other).all())


if __name__ == "__main__":
    unittest.main()
