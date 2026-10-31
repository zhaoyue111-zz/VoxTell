import math
import inspect
import unittest
from types import SimpleNamespace

import numpy as np

from voxtell.inference.oracle_support_diagnostic import (
    align_mask_nearest,
    binary_metrics,
    erode_lesions,
    exact_topk_mask,
    gaussian_fuse_patches,
    normalized_prototype,
    safe_auc,
)


class OracleSupportDiagnosticTests(unittest.TestCase):
    def test_feature_gt_alignment_uses_nearest(self):
        mask = np.zeros((2, 2, 2), dtype=bool)
        mask[0, 0, 0] = True
        aligned = align_mask_nearest(mask, (4, 4, 4))
        self.assertEqual(aligned.dtype, np.bool_)
        self.assertEqual(int(aligned.sum()), 8)
        self.assertTrue(aligned[:2, :2, :2].all())

    def test_gaussian_fusion_normalizes_overlap(self):
        gaussian = np.ones((2, 2, 2), dtype=np.float32)
        fused = gaussian_fuse_patches(
            [np.ones((2, 2, 2)), np.full((2, 2, 2), 3)],
            [
                (slice(0, 2), slice(0, 2), slice(0, 2)),
                (slice(1, 3), slice(0, 2), slice(0, 2)),
            ],
            gaussian,
            (3, 2, 2),
        )
        np.testing.assert_allclose(fused[0], 1)
        np.testing.assert_allclose(fused[1], 2)
        np.testing.assert_allclose(fused[2], 3)

    def test_prototype_is_unit_normalized_after_mean(self):
        prototype = normalized_prototype(np.asarray([[3, 0], [1, 0], [2, 0]], np.float32))
        np.testing.assert_allclose(prototype, [1, 0], atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(prototype)), 1.0, places=6)

    def test_topk_count_is_exact_even_with_ties(self):
        scores = np.ones((3, 3, 3), dtype=np.float32)
        roi = np.ones_like(scores, dtype=bool)
        selected = exact_topk_mask(scores, roi, 7)
        self.assertEqual(int(selected.sum()), 7)
        np.testing.assert_array_equal(np.flatnonzero(selected), np.arange(7))

    def test_empty_mask_and_single_class_auc_do_not_raise(self):
        empty = np.zeros((2, 2, 2), dtype=bool)
        metrics = binary_metrics(empty, empty)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["pred_foreground_voxels"], 0)
        auroc, auprc, reason = safe_auc(empty, np.zeros_like(empty, dtype=float))
        self.assertTrue(math.isnan(auroc))
        self.assertTrue(math.isnan(auprc))
        self.assertEqual(reason, "single_class_no_positive")
        auroc, auprc, reason = safe_auc(np.asarray([], bool), np.asarray([], float))
        self.assertTrue(math.isnan(auroc))
        self.assertTrue(math.isnan(auprc))
        self.assertEqual(reason, "empty_evaluation_region")

    def test_one_voxel_lesion_erosion_falls_back(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        eroded, records = erode_lesions(mask)
        self.assertTrue(np.array_equal(eroded, mask))
        self.assertTrue(records[0]["erosion_fallback"])

    def test_return_features_is_opt_in(self):
        from voxtell.model.voxtell_model import VoxTellDecoder, VoxTellModel

        model_parameter = inspect.signature(VoxTellModel.forward).parameters["return_features"]
        decoder_parameter = inspect.signature(VoxTellDecoder.forward).parameters["return_features"]
        self.assertIs(model_parameter.default, False)
        self.assertIs(decoder_parameter.default, False)

    def test_decoder_returns_identical_logits_with_feature_opt_in(self):
        import torch
        from torch import nn
        from voxtell.model.voxtell_model import VoxTellDecoder

        encoder = SimpleNamespace(
            output_channels=[2] * 6,
            conv_op=nn.Conv3d,
            conv_bias=True,
            norm_op=None,
            norm_op_kwargs=None,
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": False},
            kernel_sizes=[3] * 6,
            strides=[1, 2, 2, 2, 2, 2],
        )
        decoder = VoxTellDecoder(
            encoder=encoder,
            num_classes=1,
            n_conv_per_stage=[1] * 5,
            deep_supervision=False,
            num_maskformer_stages=5,
            num_heads=1,
        ).eval()
        skips = [
            torch.randn(1, 2, size, size, size)
            for size in (32, 16, 8, 4, 2, 1)
        ]
        embeddings = [torch.randn(1, 1, 2) for _ in range(5)]
        with torch.no_grad():
            default = decoder(skips, embeddings)
            opted_in, d5 = decoder(skips, embeddings, return_features=True)
        self.assertEqual(tuple(d5.shape), (1, 2, 32, 32, 32))
        torch.testing.assert_close(default[0], opted_in[0])


if __name__ == "__main__":
    unittest.main()
