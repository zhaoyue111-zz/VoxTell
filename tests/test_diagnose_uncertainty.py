import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from voxtell.inference.diagnose_uncertainty import (
    bbox_to_slices,
    binary_dice_from_crop,
    classification_metrics,
    correlation,
    load_cue_scores,
    mean_pairwise_soft_dice,
    mean_voxel_variance,
    _json_safe,
    weak_augment,
)


class TestDiagnoseUncertainty(unittest.TestCase):
    def test_pairwise_soft_dice_and_variance_have_expected_direction(self):
        stable = [np.ones((2, 2), dtype=np.float32)] * 3
        self.assertEqual(mean_pairwise_soft_dice(stable), 1.0)
        self.assertEqual(mean_voxel_variance(stable), 0.0)

        unstable = [
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 1.0]], dtype=np.float32),
        ]
        self.assertEqual(mean_pairwise_soft_dice(unstable), 0.0)
        self.assertEqual(mean_voxel_variance(unstable), 0.25)

    def test_binary_dice_counts_gt_outside_voxtell_crop_as_false_negative(self):
        target = np.zeros((3, 3, 3), dtype=bool)
        target[1, 1, 1] = True
        target[2, 2, 2] = True
        crop = bbox_to_slices(((0, 2), (0, 2), (0, 2)), target.shape)
        probability = np.zeros((2, 2, 2), dtype=np.float32)
        probability[1, 1, 1] = 1.0
        self.assertEqual(binary_dice_from_crop(probability, target, crop, 0.5), 2.0 / 3.0)

    def test_weak_augment_is_seeded_and_does_not_change_geometry(self):
        image = torch.linspace(-2, 2, 60, dtype=torch.float32).reshape(1, 3, 4, 5)
        first = weak_augment(image, np.random.default_rng(7))
        second = weak_augment(image, np.random.default_rng(7))
        self.assertEqual(first.shape, image.shape)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, image))

    def test_cue_csv_join_preserves_raw_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cue.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("filename", "quality_score"))
                writer.writeheader()
                writer.writerow({"filename": "case_1.nii.gz", "quality_score": "0.75"})
            scores, case_column, score_column = load_cue_scores(path, None, None)
        self.assertEqual(scores, {"case_1": 0.75})
        self.assertEqual(case_column, "filename")
        self.assertEqual(score_column, "quality_score")

    def test_statistics_use_negative_uncertainty_as_quality_score(self):
        uncertainty = [0.1, 0.2, 0.8, 0.9]
        dice = [0.9, 0.8, 0.3, 0.1]
        pearson, count = correlation(uncertainty, dice)
        spearman, _ = correlation(uncertainty, dice, rank=True)
        auroc, auprc, classification_count = classification_metrics(uncertainty, dice, 0.7)
        self.assertEqual(count, classification_count)
        self.assertEqual(count, 4)
        self.assertLess(pearson, 0)
        self.assertEqual(spearman, -1.0)
        self.assertEqual(auroc, 1.0)
        self.assertEqual(auprc, 1.0)

    def test_json_safe_converts_numpy_scalars_and_nonfinite_values(self):
        self.assertEqual(_json_safe({"count": np.int64(2)}), {"count": 2})
        self.assertEqual(_json_safe(np.float32(0.25)), 0.25)
        self.assertIsNone(_json_safe(np.float64(np.nan)))


if __name__ == "__main__":
    unittest.main()
