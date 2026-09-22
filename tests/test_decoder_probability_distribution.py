import math
import unittest

import numpy as np

from voxtell.inference.decoder_probability_distribution import (
    align_gt_nearest,
    align_probability_trilinear,
    analyze_case,
    decoder_metadata,
    probability_stats,
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
                         {"gt_foreground", "gt_background", "d5_false_negative"})
        d5_fn = [row for row in rows if row["decoder_stage"] == "D5"
                 and row["region"] == "d5_false_negative"][0]
        self.assertEqual(d5_fn["voxel_count"], 1)
        self.assertAlmostEqual(d5_fn["mean"], 0.2)
        background = [row for row in rows if row["decoder_stage"] == "D5"
                      and row["region"] == "gt_background"][0]
        self.assertEqual(background["voxel_count"], 1)
        self.assertAlmostEqual(background["mean"], 0.3)
        self.assertEqual(rows[0]["aligned_shape"], "[1, 1, 2]")

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
