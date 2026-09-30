import unittest

import torch

from utils.shadow_transport import apply_sensitivity_anchor as _apply_sensitivity_anchor


class _Model:
    def __init__(self, anchor):
        self.texture_shadow_sensitivity_anchor = anchor


class SensitivityAnchorTests(unittest.TestCase):
    def setUp(self):
        self.composed = torch.tensor([0.20, 0.50, 0.80])
        self.uv = torch.tensor([0.40, 0.60, 0.90])

    def _apply(self, anchor, sensitivity):
        return _apply_sensitivity_anchor(
            self.composed, self.uv, torch.as_tensor(sensitivity), _Model(anchor)
        )

    def test_no_shadow_anchor_leaves_zero_sensitivity_fully_lit(self):
        torch.testing.assert_close(self._apply("no_shadow", torch.zeros(3)),
                                   torch.ones(3))

    def test_visibility_anchor_leaves_zero_sensitivity_at_uv(self):
        torch.testing.assert_close(self._apply("visibility", torch.zeros(3)),
                                   self.uv)

    def test_full_sensitivity_reaches_the_range_value_under_both_anchors(self):
        for anchor in ("no_shadow", "visibility"):
            with self.subTest(anchor=anchor):
                torch.testing.assert_close(self._apply(anchor, torch.ones(3)),
                                           self.composed)

    def test_anchors_differ_at_partial_sensitivity(self):
        half = torch.full((3,), 0.5)
        lit = self._apply("no_shadow", half)
        vis = self._apply("visibility", half)
        torch.testing.assert_close(lit, 1.0 + 0.5 * (self.composed - 1.0))
        torch.testing.assert_close(vis, self.uv + 0.5 * (self.composed - self.uv))
        self.assertFalse(torch.allclose(lit, vis))

    def test_no_shadow_anchor_is_the_default(self):
        class Bare:
            pass

        torch.testing.assert_close(
            _apply_sensitivity_anchor(self.composed, self.uv, torch.zeros(3), Bare()),
            torch.ones(3),
        )

    def test_output_stays_in_unit_range(self):
        for anchor in ("no_shadow", "visibility"):
            for level in (0.0, 0.25, 0.5, 0.75, 1.0):
                out = self._apply(anchor, torch.full((3,), level))
                self.assertTrue(bool((out >= 0.0).all()) and bool((out <= 1.0).all()))

    def test_unknown_anchor_is_rejected(self):
        with self.assertRaises(ValueError):
            self._apply("halfway", torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
