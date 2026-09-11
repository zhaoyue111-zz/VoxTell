import math
from pathlib import Path
import unittest

import torch

from voxtell.inference.text_causal_quality import (
    _safe_padding_mask,
    _attention_stats,
    accumulate_causal_patch_logits,
    aggregate_attention_coverages,
    build_memory_masks,
    crop_global_mask_to_patch,
    fused_prediction_softdices,
    attention_coverage,
    soft_dice,
    UNSUPERVISED_SCORES,
)
from voxtell.model.transformer import TransformerDecoderLayer


class TextCausalQualityTest(unittest.TestCase):
    def _make_tiny_voxtell(self):
        try:
            from dynamic_network_architectures.building_blocks.residual import BasicBlockD
            from voxtell.model.voxtell_model import VoxTellModel
        except ModuleNotFoundError as exc:
            self.skipTest(f"full VoxTell model dependencies unavailable: {exc}")
        from torch import nn

        old_configs = VoxTellModel.DECODER_CONFIGS
        VoxTellModel.DECODER_CONFIGS = {
            0: {"channels": 4, "shape": (8, 8, 8)},
            1: {"channels": 4, "shape": (4, 4, 4)},
        }
        try:
            return VoxTellModel(
                input_channels=1, n_stages=2, features_per_stage=[4, 4],
                conv_op=nn.Conv3d, kernel_sizes=[3, 3], strides=[1, 2],
                n_blocks_per_stage=1, n_conv_per_stage_decoder=[1],
                conv_bias=False, norm_op=nn.InstanceNorm3d,
                norm_op_kwargs={"eps": 1e-5, "affine": True},
                dropout_op=None, dropout_op_kwargs=None,
                nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
                deep_supervision=False, block=BasicBlockD,
                num_maskformer_stages=1, query_dim=8, decoder_layer=0,
                text_embedding_dim=4, num_heads=1,
                project_to_decoder_hidden_dim=4,
            )
        finally:
            VoxTellModel.DECODER_CONFIGS = old_configs

    def test_real_forward_default_d5_matches_diagnostic_forward(self):
        model = self._make_tiny_voxtell().eval()
        image = torch.randn(1, 1, 8, 8, 8)
        text = torch.randn(1, 1, 4)
        with torch.no_grad():
            plain = model(image, text)
            diagnostic_pred, diagnostics = model(
                image, text, return_diagnostics=True
            )
        self.assertIsInstance(plain, torch.Tensor)
        self.assertIsInstance(diagnostic_pred, torch.Tensor)
        self.assertEqual(tuple(plain.shape), tuple(diagnostic_pred.shape))
        self.assertTrue(torch.equal(plain, diagnostic_pred))
        self.assertEqual(tuple(diagnostics["q"].shape), (1, 1, 8))
        self.assertEqual(diagnostics["q"].device, image.device)

    def test_cuda_diagnostic_q_device_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        model = self._make_tiny_voxtell().to("cuda").eval()
        image = torch.randn(1, 1, 8, 8, 8, device="cuda")
        text = torch.randn(1, 1, 4, device="cuda")
        with torch.no_grad():
            _, diagnostics = model(image, text, return_diagnostics=True)
        self.assertEqual(diagnostics["q"].device, image.device)

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
        mean, weighted, median = aggregate_attention_coverages(
            [0.2, 0.8], [1, 3]
        )
        self.assertAlmostEqual(mean, 0.5)
        self.assertAlmostEqual(weighted, 0.65)
        self.assertAlmostEqual(median, 0.5)

    def test_attention_stats_measure_total_mass_not_mean_per_token(self):
        attention_a = torch.tensor([[[0.1, 0.1, 0.4, 0.4]]])
        blocked_a = torch.tensor([[True, True, False, False]])
        blocked_mass_a, visible_mass_a = _attention_stats(attention_a, blocked_a)
        self.assertAlmostEqual(blocked_mass_a, 0.2, places=6)
        self.assertAlmostEqual(visible_mass_a, 0.8, places=6)

        attention_b = torch.tensor([[[0.1, 0.1, 0.2, 0.2, 0.2, 0.2]]])
        blocked_b = torch.tensor([[True, True, False, False, False, False]])
        blocked_mass_b, visible_mass_b = _attention_stats(attention_b, blocked_b)
        self.assertAlmostEqual(blocked_mass_b, 0.2, places=6)
        self.assertAlmostEqual(visible_mass_b, 0.8, places=6)
        self.assertAlmostEqual(visible_mass_a, visible_mass_b, places=6)

    def test_attention_stats_support_multihead_attention(self):
        attention = torch.tensor([
            [
                [[0.1, 0.2, 0.3, 0.4]],
                [[0.2, 0.1, 0.2, 0.5]],
            ]
        ])
        blocked = torch.tensor([[True, True, False, False]])
        blocked_mass, visible_mass = _attention_stats(attention, blocked)
        self.assertAlmostEqual(blocked_mass, 0.3, places=6)
        self.assertAlmostEqual(visible_mass, 0.7, places=6)

    def test_overlapping_windows_share_causal_normal_in_out_support(self):
        shape = (1, 4, 4)
        causal_normal = torch.zeros(shape)
        in_sum = torch.zeros(shape)
        out_sum = torch.zeros(shape)
        causal_denominator = torch.zeros(shape)
        first = (slice(0, 1), slice(0, 3), slice(0, 3))
        second = (slice(0, 1), slice(1, 4), slice(1, 4))
        ones = torch.ones((1, 3, 3))
        accumulate_causal_patch_logits(
            causal_normal, in_sum, out_sum, causal_denominator,
            ones, ones * 2, ones * 3, first, 1.0, True,
        )
        accumulate_causal_patch_logits(
            causal_normal, in_sum, out_sum, causal_denominator,
            ones * 10, ones * 20, ones * 30, second, 1.0, False,
        )
        valid = causal_denominator > 0
        self.assertEqual(int(valid.sum()), 9)
        self.assertTrue(torch.equal(causal_denominator[valid], torch.ones(9)))
        self.assertTrue(torch.equal(causal_normal[valid], torch.ones(9)))
        self.assertTrue(torch.equal(in_sum[valid], torch.full((9,), 2.0)))
        self.assertTrue(torch.equal(out_sum[valid], torch.full((9,), 3.0)))
        self.assertFalse(bool(valid[:, 3, 3]))

    def test_attention_scores_are_included_in_correlation_inputs(self):
        for name in (
            "q_similarity_in_median", "q_similarity_out_median",
            "attention_coverage_normal", "attention_coverage_normal_weighted",
            "attention_coverage_normal_median",
        ):
            self.assertIn(name, UNSUPERVISED_SCORES)

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
        support = torch.tensor([True, False])
        supported_in, supported_out = fused_prediction_softdices(
            normal, inside, outside, 1, support
        )
        self.assertAlmostEqual(
            supported_in,
            float(soft_dice(torch.sigmoid(normal[support]), torch.sigmoid(inside[support]))),
        )
        self.assertAlmostEqual(
            supported_out,
            float(soft_dice(torch.sigmoid(normal[support]), torch.sigmoid(outside[support]))),
        )


if __name__ == "__main__":
    unittest.main()
