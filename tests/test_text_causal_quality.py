import math
from pathlib import Path
import unittest

import torch

from voxtell.inference.text_causal_quality import (
    _safe_padding_mask,
    build_memory_masks,
    crop_global_mask_to_patch,
    fused_prediction_softdices,
    attention_coverage,
    soft_dice,
)
from voxtell.model.transformer import TransformerDecoderLayer


class TextCausalQualityTest(unittest.TestCase):
    def test_default_d5_interface_path_is_unchanged(self):
        source = (Path(__file__).parents[1] / "voxtell/model/voxtell_model.py").read_text()
        self.assertIn("memory_key_padding_mask: Optional[torch.Tensor] = None", source)
        self.assertIn("return_diagnostics: bool = False", source)
        self.assertIn("if not self.deep_supervision and not return_decoder_outputs:", source)
        self.assertIn("outs = outs[0]", source)
        self.assertIn('"q": mask_embedding', source)

    def test_memory_mask_alignment_uses_hwd_flatten_order(self):
        pseudo = torch.zeros(1, 2, 2, 2)
        pseudo[0, 1, 0, 1] = 1  # d=1,h=0,w=1 -> HWD token index 3
        masks = build_memory_masks(pseudo, (2, 2, 2))
        self.assertEqual(tuple(masks["inside_tokens"].shape), (1, 8))
        self.assertTrue(bool(masks["inside_tokens"][0, 3]))
        self.assertEqual(int(masks["inside_tokens"].sum()), 1)

    def test_global_mask_crop_matches_sliding_window_coordinates(self):
        global_mask = torch.zeros(1, 4, 5, 6, dtype=torch.bool)
        global_mask[0, 2, 3, 4] = True
        slicer = (slice(None), slice(1, 4), slice(2, 5), slice(3, 6))
        patch = crop_global_mask_to_patch(global_mask, slicer)
        self.assertEqual(tuple(patch.shape), (1, 3, 3, 3))
        self.assertTrue(bool(patch[0, 1, 1, 1]))

    def test_empty_and_full_masks_are_safe(self):
        empty = build_memory_masks(torch.zeros(1, 2, 2, 2), (2, 2, 2))
        full = build_memory_masks(torch.ones(1, 2, 2, 2), (2, 2, 2))
        self.assertEqual(empty["fallback_in"], 1)
        self.assertEqual(full["fallback_out"], 1)
        for masks in (empty, full):
            for key in ("only_in", "only_out"):
                self.assertFalse(bool(torch.isnan(masks[key].float()).any()))
                self.assertFalse(bool(masks[key].all(dim=1).any()))

    def test_masked_attention_is_zero_and_no_nan(self):
        torch.manual_seed(0)
        layer = TransformerDecoderLayer(
            d_model=8, nhead=2, dropout=0.0, normalize_before=True
        ).eval()
        tgt = torch.randn(1, 1, 8)
        memory = torch.randn(4, 1, 8)
        padding = torch.tensor([[False, True, True, False]])
        with torch.no_grad():
            output, attention = layer(
                tgt, memory, memory_key_padding_mask=padding
            )
        self.assertFalse(bool(torch.isnan(output).any()))
        self.assertFalse(bool(torch.isnan(attention).any()))
        self.assertTrue(torch.allclose(attention[..., 1:3], torch.zeros_like(attention[..., 1:3]), atol=1e-7))

    def test_normal_attention_coverage(self):
        attention = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]])
        inside = torch.tensor([[False, True, False, True]])
        self.assertAlmostEqual(attention_coverage(attention, inside), 0.6)

    def test_all_masked_attention_has_no_nan(self):
        layer = TransformerDecoderLayer(
            d_model=8, nhead=2, dropout=0.0, normalize_before=True
        ).eval()
        tgt = torch.randn(1, 1, 8)
        memory = torch.randn(4, 1, 8)
        safe, fallback = _safe_padding_mask(torch.ones(1, 4, dtype=torch.bool))
        self.assertEqual(fallback, 1)
        with torch.no_grad():
            output, attention = layer(
                tgt, memory, memory_key_padding_mask=safe
            )
        self.assertFalse(bool(torch.isnan(output).any()))
        self.assertFalse(bool(torch.isnan(attention).any()))

    def test_soft_dice_is_finite_for_empty_predictions(self):
        score = soft_dice(torch.zeros(4), torch.zeros(4))
        self.assertTrue(math.isfinite(float(score)))
        self.assertEqual(float(score), 1.0)

    def test_fallback_patch_is_excluded_and_fused_softdice_uses_whole_volume(self):
        empty = build_memory_masks(torch.zeros(1, 2, 2, 2), (2, 2, 2))
        self.assertEqual(empty["fallback_in"], 1)
        self.assertEqual(empty["fallback_out"], 0)
        normal = torch.tensor([0.0, 0.0])
        inside = torch.tensor([0.0, 2.0])
        outside = torch.tensor([2.0, 0.0])
        score_in, score_out = fused_prediction_softdices(normal, inside, outside, 1)
        expected_in = float(soft_dice(torch.sigmoid(normal), torch.sigmoid(inside)))
        expected_out = float(soft_dice(torch.sigmoid(normal), torch.sigmoid(outside)))
        self.assertAlmostEqual(score_in, expected_in)
        self.assertAlmostEqual(score_out, expected_out)
        invalid = fused_prediction_softdices(normal, inside, outside, 0)
        self.assertTrue(math.isnan(invalid[0]))
        self.assertTrue(math.isnan(invalid[1]))


if __name__ == "__main__":
    unittest.main()
