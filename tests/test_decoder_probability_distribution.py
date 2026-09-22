import math
import unittest

import numpy as np

from voxtell.inference.decoder_probability_distribution import (
    align_gt_nearest,
    align_probability_trilinear,
    analyze_case,
    decoder_metadata,
    probability_stats,
    select_label_value,
    summarize_rows,
)


class DecoderProbabilityDistributionTest(unittest.TestCase):
    def test_decoder_mapping_uses_actual_returned_list_indices(self):
        metadata = decoder_metadata(
            [(192, 192, 192), (96, 96, 96), (48, 48, 48), (24, 24, 24), (12, 12, 12)],
            (192, 192, 192),
        )
        self.assertEqual([item["decoder_stage"] for item in metadata],
                         ["D1", "D2", "D3", "D4", "D5"])
        self.assertEqual([item["model_output_list_index"] for item in metadata], [4, 3, 2, 1, 0])
        self.assertEqual(metadata[0]["raw_shape"], [12, 12, 12])
        self.assertTrue(metadata[-1]["is_final_output"])

    def test_interpolation_and_region_partition(self):
        gt = np.zeros((2, 2, 2), dtype=bool)
        gt[0, 0, 0] = True
        self.assertEqual(align_gt_nearest(gt.astype(np.float32), (4, 4, 4)).shape, (4, 4, 4))
        self.assertEqual(align_probability_trilinear(np.ones((1, 1, 1), np.float32), (4, 4, 4)).shape,
                         (4, 4, 4))

        # Returned order is D5, D4, D3, D2, D1.  D5 misses one GT voxel and
        # the second voxel is an explicit GT-background voxel.
        returned = [np.array([[[value, value + 0.1]]], dtype=np.float32)
                    for value in (0.2, 0.3, 0.4, 0.6, 0.8)]
        rows = analyze_case(returned, np.array([[[True, False]]]), case="case")
        self.assertEqual({row["region"] for row in rows},
                         {"gt_foreground", "gt_background_valid", "gt_background_all",
                          "d5_false_negative"})
        d5_fn = [row for row in rows if row["decoder_stage"] == "D5"
                 and row["region"] == "d5_false_negative"][0]
        self.assertEqual(d5_fn["voxel_count"], 1)
        self.assertAlmostEqual(d5_fn["mean"], 0.2)
        background = [row for row in rows if row["decoder_stage"] == "D5"
                      and row["region"] == "gt_background_valid"][0]
        self.assertEqual(background["voxel_count"], 1)
        self.assertAlmostEqual(background["mean"], 0.3)
        self.assertEqual(rows[0]["aligned_shape"], "[1, 1, 2]")

    def test_crop_outside_zero_probabilities_are_excluded_from_valid_background(self):
        gt = np.zeros((1, 1, 4), dtype=bool)
        valid = np.array([[[True, True, False, False]]])
        returned = [np.array([[[0.2, 0.4, 0.0, 0.0]]], dtype=np.float32)] * 5
        rows = analyze_case(returned, gt, case="cropped", valid_inference_mask=valid)
        valid_background = next(row for row in rows
                                if row["decoder_stage"] == "D5"
                                and row["region"] == "gt_background_valid")
        all_background = next(row for row in rows
                              if row["decoder_stage"] == "D5"
                              and row["region"] == "gt_background_all")
        self.assertEqual(valid_background["voxel_count"], 2)
        self.assertAlmostEqual(valid_background["mean"], 0.3)
        self.assertEqual(all_background["voxel_count"], 4)
        self.assertAlmostEqual(all_background["mean"], 0.15)
        self.assertTrue(all(row["region"] != "gt_background_all"
                            for row in summarize_rows(rows)))

    def test_label_selection_rejects_ambiguous_nonzero_labels(self):
        self.assertEqual(select_label_value(np.array([[[0, 7, 7]]]), None, "one"), 7)
        with self.assertRaisesRegex(ValueError, r"\[3, 7\].*--label-value"):
            select_label_value(np.array([[[0, 3, 7]]]), None, "ambiguous")

    def test_real_decoder_forward_order_matches_metadata(self):
        try:
            from dynamic_network_architectures.building_blocks.residual import BasicBlockD
            from torch import nn
            import torch
            from voxtell.model.voxtell_model import VoxTellModel
        except ModuleNotFoundError as exc:
            self.skipTest(f"full VoxTell model dependencies unavailable: {exc}")

        old_configs = VoxTellModel.DECODER_CONFIGS
        VoxTellModel.DECODER_CONFIGS = {
            i: {"channels": 2, "shape": (32 // (2 ** i),) * 3}
            for i in range(6)
        }
        try:
            model = VoxTellModel(
                input_channels=1, n_stages=6, features_per_stage=[2] * 6,
                conv_op=nn.Conv3d, kernel_sizes=[3] * 6,
                strides=[1, 2, 2, 2, 2, 2], n_blocks_per_stage=[1] * 6,
                n_conv_per_stage_decoder=[1] * 5, conv_bias=False,
                norm_op=nn.InstanceNorm3d,
                norm_op_kwargs={"eps": 1e-5, "affine": True},
                dropout_op=None, dropout_op_kwargs=None,
                nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
                deep_supervision=True, block=BasicBlockD,
                num_maskformer_stages=5, query_dim=8, decoder_layer=5,
                text_embedding_dim=4, num_heads=1,
                project_to_decoder_hidden_dim=4,
            ).eval()
            with torch.no_grad():
                outputs = model(
                    torch.randn(1, 1, 32, 32, 32), torch.randn(1, 1, 4),
                    return_decoder_outputs=True,
                )
            shapes = [tuple(output.shape[2:]) for output in outputs]
            metadata = model.decoder.get_output_metadata(
                observed_output_shapes=shapes,
                patch_spatial_shape=(32, 32, 32),
            )
            self.assertEqual(metadata[-1]["model_output_list_index"], 0)
            self.assertTrue(metadata[-1]["is_final_output"])
            self.assertEqual(metadata[0]["model_output_list_index"], 4)
            self.assertGreater(np.prod(shapes[0]), np.prod(shapes[-1]))
            self.assertEqual(shapes[0], tuple(metadata[-1]["observed_raw_patch_shape"]))
            self.assertEqual(shapes[-1], tuple(metadata[0]["observed_raw_patch_shape"]))
        finally:
            VoxTellModel.DECODER_CONFIGS = old_configs

    def test_quantiles_and_empty_region(self):
        stats = probability_stats(np.array([0, 1, 2, 3, 4], dtype=np.float32))
        self.assertEqual(stats["voxel_count"], 5)
        self.assertAlmostEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["p05"], 0.2)
        self.assertAlmostEqual(stats["median"], 2.0)
        empty = probability_stats(np.array([], dtype=np.float32))
        self.assertEqual(empty["voxel_count"], 0)
        self.assertTrue(math.isnan(empty["mean"]))

    def test_summary_is_mean_of_case_statistics(self):
        rows = []
        for case, value in (("small", 0.1), ("large", 0.9)):
            rows.extend(analyze_case(
                [np.array([[[value]]], dtype=np.float32)] * 5,
                np.ones((1, 1, 1), dtype=bool),
                case=case,
            ))
        summary = summarize_rows(rows)
        row = next(item for item in summary
                   if item["decoder_stage"] == "D5" and item["region"] == "gt_foreground")
        self.assertEqual(row["case_count"], 2)
        self.assertAlmostEqual(row["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
