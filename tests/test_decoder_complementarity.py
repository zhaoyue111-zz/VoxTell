import math
import unittest

import numpy as np

from voxtell.inference.decoder_complementarity import (
    analyze_case,
    binary_metrics,
    majority_vote,
    overlap_metrics,
)


class DecoderComplementarityTest(unittest.TestCase):
    def setUp(self):
        self.gt = np.array([[[1, 1, 0, 0, 0, 0]]], dtype=bool)
        self.probs = [np.zeros_like(self.gt, dtype=np.float32) for _ in range(5)]
        # D1 has one genuinely new voxel and one false-positive voxel.
        self.probs[0][0, 0, 1] = .9
        self.probs[0][0, 0, 2] = .9
        # D2--D5 agree on the first GT voxel; D5 is the final output.
        for p in self.probs[1:]:
            p[0, 0, 0] = .9

    def test_basic_metrics_and_empty_denominator(self):
        m = binary_metrics(self.probs[0] >= .5, self.gt)
        self.assertEqual(m["intersection_voxels"], 1)
        self.assertAlmostEqual(m["precision"], .5)
        self.assertAlmostEqual(m["recall"], .5)
        empty = binary_metrics(np.zeros_like(self.gt), np.zeros_like(self.gt))
        self.assertTrue(math.isnan(empty["dice"]))
        self.assertTrue(math.isnan(empty["precision"]))

    def test_extra_precision_fn_recovery_and_merge(self):
        result = analyze_case(self.probs, self.gt)
        row = next(r for r in result["rows"] if r.get("record_type") == "decoder1_extra")
        self.assertEqual(row["r1_voxels"], 2)
        self.assertEqual(row["r1_tp_voxels"], 1)
        self.assertEqual(row["r1_fp_voxels"], 1)
        self.assertAlmostEqual(row["extra_precision"], .5)
        self.assertAlmostEqual(row["fn_recovery"], 1.)
        self.assertEqual(row["merge_added_tp_voxels"], 1)
        self.assertEqual(row["merge_added_fp_voxels"], 1)
        self.assertAlmostEqual(row["merge_precision"], 2 / 3)

    def test_overlap_union_intersection_and_majority(self):
        a = np.array([1, 1, 0], dtype=bool)
        b = np.array([1, 0, 1], dtype=bool)
        overlap = overlap_metrics(a, b)
        self.assertEqual(overlap["intersection_voxels"], 1)
        self.assertEqual(overlap["union_voxels"], 3)
        self.assertAlmostEqual(overlap["overlap"], 1 / 3)
        self.assertTrue(np.array_equal(majority_vote([a, b, a], 2), np.array([1, 1, 0])))
        pair = next(r for r in analyze_case(self.probs, self.gt)["rows"] if r.get("record_type") == "pairwise")
        self.assertEqual(pair["union_voxels"], 0)  # D2--D5 FP regions are empty
        self.assertTrue(math.isnan(pair["fp_overlap"]))
        ensembles = [r for r in analyze_case(self.probs, self.gt)["rows"] if r.get("record_type") == "ensemble"]
        intersection = next(r for r in ensembles if r["name"] == "d2_d5_intersection")
        union = next(r for r in ensembles if r["name"] == "d2_d5_union")
        majority = next(r for r in ensembles if r["name"] == "d2_d5_majority_vote")
        self.assertEqual(intersection["d25_intersection_voxels"], 1)
        self.assertEqual(union["d25_union_voxels"], 1)
        self.assertEqual(majority["majority_vote_voxels"], 1)


if __name__ == "__main__":
    unittest.main()
