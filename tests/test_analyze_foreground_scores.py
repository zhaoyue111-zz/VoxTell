import math
import unittest

import numpy as np

from voxtell.inference.analyze_foreground_scores import (
    analyze_probability_volume,
    exact_top_fraction_mask,
    foreground_probability_stats,
    parse_args,
    select_axial_slices,
    summarize_cases,
)


class AnalyzeForegroundScoresTest(unittest.TestCase):
    def test_cli_accepts_one_image_and_one_label(self):
        args = parse_args(["--image", "case.nii.gz", "--label", "case_gt.nii.gz"])
        self.assertEqual(args.image, "case.nii.gz")
        self.assertEqual(args.label, "case_gt.nii.gz")
        self.assertFalse(hasattr(args, "limit"))

    def test_foreground_statistics_and_strict_cutoffs(self):
        values = np.array([0.0, 0.1, 0.2, 0.5, 1.0], dtype=np.float32)
        stats = foreground_probability_stats(values)
        self.assertEqual(stats["voxel_count"], 5)
        self.assertAlmostEqual(stats["mean"], 0.36)
        self.assertAlmostEqual(stats["median"], 0.2)
        self.assertAlmostEqual(stats["p01"], 0.004)
        self.assertAlmostEqual(stats["probability_gt_0_1_fraction"], 3 / 5)
        self.assertAlmostEqual(stats["probability_gt_0_5_fraction"], 1 / 5)

    def test_exact_top_five_percent_excludes_invalid_and_resolves_ties(self):
        # 21 valid voxels require ceil(1.05) == 2 selections. All scores tie,
        # so a percentile comparison would incorrectly select all 21.
        probability = np.full((1, 1, 22), 0.7, dtype=np.float32)
        valid = np.ones_like(probability, dtype=bool)
        valid[..., -1] = False
        probability[..., -1] = 1.0
        mask, boundary = exact_top_fraction_mask(probability, valid)
        self.assertEqual(int(mask.sum()), 2)
        self.assertEqual(int(mask[..., -1].sum()), 0)
        self.assertAlmostEqual(boundary, 0.7)

    def test_analysis_localization_fields(self):
        probability = np.arange(20, dtype=np.float32).reshape(2, 2, 5) / 20
        valid = np.ones_like(probability, dtype=bool)
        gt = np.zeros_like(probability, dtype=bool)
        gt[1, 1, 4] = True
        result, top5, max_zyx = analyze_probability_volume(
            probability, gt, valid, threshold=0.5
        )
        self.assertEqual(result["voxel_count"], 1)
        self.assertEqual(result["gt_voxel_count"], 1)
        self.assertEqual(result["pred_voxel_count"], 9)
        self.assertFalse(result["empty_prediction"])
        self.assertEqual(result["top5_voxel_count"], math.ceil(0.05 * 20))
        self.assertEqual(result["top5_in_gt_count"], 1)
        self.assertEqual(result["global_max_x"], 4)
        self.assertEqual(result["global_max_y"], 1)
        self.assertEqual(result["global_max_z"], 1)
        self.assertEqual(result["global_max_xyz"], "(4,1,1)")
        self.assertEqual(max_zyx, (1, 1, 4))
        self.assertEqual(int(top5.sum()), 1)

    def test_empty_gt_is_an_error(self):
        shape = (2, 2, 2)
        with self.assertRaisesRegex(ValueError, "GT foreground is empty"):
            analyze_probability_volume(
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=bool),
                np.ones(shape, dtype=bool),
                threshold=0.5,
            )

    def test_axial_selection_always_includes_global_max_slice(self):
        mask = np.zeros((8, 2, 2), dtype=np.uint8)
        mask[:6] = 1
        selected = select_axial_slices(mask, (7, 0, 0), maximum_slices=3)
        self.assertEqual(selected[:3], [0, 1, 2])
        self.assertIn(7, selected)

    def test_summary_uses_micro_overlap_counts(self):
        rows = [
            {
                "empty_prediction": True, "threshold": 0.5, "prompt": "liver",
                "valid_voxel_count": 100, "gt_voxel_count": 10,
                "pred_voxel_count": 0, "top5_voxel_count": 5,
                "top5_in_gt_count": 1, "mean": 0.1,
                "gt_max_probability": 0.2, "gt_fraction_above_threshold": 0.0,
                "top5_precision": 0.2, "gt_covered_by_top5": 0.1,
            },
            {
                "empty_prediction": False, "threshold": 0.5, "prompt": "liver",
                "valid_voxel_count": 200, "gt_voxel_count": 30,
                "pred_voxel_count": 2, "top5_voxel_count": 10,
                "top5_in_gt_count": 5, "mean": 0.3,
                "gt_max_probability": 0.8, "gt_fraction_above_threshold": 0.2,
                "top5_precision": 0.5, "gt_covered_by_top5": 1 / 6,
            },
        ]
        summary = summarize_cases(rows)
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["empty_prediction_count"], 1)
        self.assertAlmostEqual(summary["micro_top5_precision"], 6 / 15)
        self.assertAlmostEqual(summary["micro_gt_covered_by_top5"], 6 / 40)


if __name__ == "__main__":
    unittest.main()
