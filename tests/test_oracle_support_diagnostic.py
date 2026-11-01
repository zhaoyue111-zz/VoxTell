import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import nibabel as nib
import torch

from voxtell.inference.oracle_support_full_volume import (
    CaseFiles,
    _load_nifti,
    assert_support_query_disjoint,
    ensure_work_dir_capacity,
    estimate_peak_work_bytes,
    effective_empty_recovery,
    fuse_probability,
    gaussian_fuse_features_before_similarity,
    paired_comparison,
    random_seed_summary,
    restore_crop_to_full,
    run_full_volume_feature_inference,
    run_query_case,
    save_nifti_like,
    shuffled_prototype_bank,
    summarize,
    verify_baseline_dice,
)


class OracleSupportDiagnosticTests(unittest.TestCase):
    def test_full_volume_crop_restoration(self):
        crop = np.arange(24).reshape(2, 3, 4)
        full = restore_crop_to_full(crop, ((1, 3), (2, 5), (1, 5)), (4, 7, 6))
        self.assertEqual(full.shape, (4, 7, 6))
        np.testing.assert_array_equal(full[1:3, 2:5, 1:5], crop)
        self.assertEqual(int(np.count_nonzero(full)), int(np.count_nonzero(crop)))

    def test_saved_nifti_preserves_original_shape_and_affine(self):
        internal = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        affine = np.array([
            [1.5, 0, 0, 10], [0, 2.0, 0, 20], [0, 0, 3.0, 30], [0, 0, 0, 1],
        ], dtype=np.float64)
        properties = {
            "nibabel_stuff": {
                "reoriented_affine": affine.copy(),
                "original_affine": affine.copy(),
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restored.nii.gz"
            save_nifti_like(internal, path, properties)
            restored = nib.load(path)
            self.assertEqual(restored.shape, (4, 3, 2))
            np.testing.assert_allclose(restored.affine, affine)

    def test_non_ras_nifti_roundtrip_preserves_voxels_shape_and_affine(self):
        original = np.arange(60, dtype=np.float32).reshape(5, 4, 3)
        affine = np.array([
            [0, -2.0, 0, 20], [-1.5, 0, 0, 10], [0, 0, 3.0, 30], [0, 0, 0, 1],
        ], dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "non_ras_source.nii.gz"
            restored_path = Path(directory) / "non_ras_restored.nii.gz"
            nib.save(nib.Nifti1Image(original, affine), source)
            internal, properties = _load_nifti(source)
            save_nifti_like(internal, restored_path, properties)

            restored = nib.load(restored_path)
            self.assertEqual(restored.shape, original.shape)
            np.testing.assert_allclose(restored.affine, affine)
            np.testing.assert_array_equal(np.asanyarray(restored.dataobj), original)
            internal_roundtrip, _ = _load_nifti(restored_path)
            np.testing.assert_array_equal(internal_roundtrip, internal)

    def test_insufficient_work_disk_fails_before_allocation(self):
        estimate = estimate_peak_work_bytes(4, (8, 8, 8), (7, 7, 7))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(OSError, "insufficient temporary disk space"):
                ensure_work_dir_capacity(Path(directory), estimate, free_bytes=estimate["peak_bytes"])

    def test_inference_exception_removes_case_work_directory(self):
        case = CaseFiles("cleanup_case", Path("image.nii.gz"), Path("label.nii.gz"), None, 0.0)
        predictor = SimpleNamespace(
            patch_size=(2, 2, 2), tile_step_size=0.5, device=torch.device("cpu"),
            network=SimpleNamespace(encoder=SimpleNamespace(output_channels=[2])),
            preprocess=lambda image: (torch.zeros((1, 2, 2, 2)), ((0, 2), (0, 2), (0, 2)), (2, 2, 2)),
            _internal_get_sliding_window_slicers=lambda shape: [
                (slice(None), slice(0, 2), slice(0, 2), slice(0, 2))
            ],
        )
        properties = {"nibabel_stuff": {"original_affine": np.eye(4)}}
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            with mock.patch(
                "voxtell.inference.oracle_support_full_volume.load_case",
                return_value=(np.ones((2, 2, 2), np.float32), np.ones((2, 2, 2), bool), properties),
            ), mock.patch(
                "voxtell.inference.oracle_support_full_volume._network_patch",
                side_effect=RuntimeError("simulated CUDA failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated CUDA failure"):
                    run_full_volume_feature_inference(
                        predictor, torch.zeros((1, 1, 1)), case, work_dir, "HCC lesion"
                    )
            self.assertFalse((work_dir / case.case_id).exists())

    def test_query_exception_runs_case_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_dir = Path(directory) / "query_case"
            temp_dir.mkdir()
            marker = temp_dir / "large_temp.dat"
            marker.write_bytes(b"temporary")
            state = {"cleaned": False}

            class FakeCase:
                def cleanup(self):
                    state["cleaned"] = True
                    if temp_dir.exists():
                        import shutil
                        shutil.rmtree(temp_dir)

            with mock.patch(
                "voxtell.inference.oracle_support_full_volume.materialize_normalized_features",
                side_effect=RuntimeError("query scoring failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "query scoring failure"):
                    run_query_case(FakeCase(), {}, {}, Path(directory), torch.device("cpu"))
            self.assertTrue(state["cleaned"])
            self.assertFalse(temp_dir.exists())

    def test_baseline_csv_mismatch_fails(self):
        self.assertLess(verify_baseline_dice("ok", 0.5, 0.50001), 1e-4)
        with self.assertRaisesRegex(RuntimeError, "baseline Dice mismatch"):
            verify_baseline_dice("bad", 0.5, 0.5002)

    def test_support_query_sets_must_not_overlap(self):
        assert_support_query_disjoint(["support"], ["query"])
        with self.assertRaisesRegex(AssertionError, "overlap"):
            assert_support_query_disjoint(["same"], ["same"])

    def test_gt_is_not_an_input_to_fusion_or_parameter_selection(self):
        parameters = inspect.signature(fuse_probability).parameters
        self.assertEqual(list(parameters), ["baseline_probability", "standardized_score", "alpha"])
        baseline = np.full((2, 2, 2), 0.4, np.float32)
        score = np.ones_like(baseline)
        np.testing.assert_array_equal(
            fuse_probability(baseline, score, 0.5),
            fuse_probability(baseline, score, 0.5),
        )

    def test_query_fuses_features_before_similarity(self):
        # At the overlap voxel patch-score averaging would yield (1+0)/2=0.5.
        # Feature fusion produces [1,1], whose cosine to [1,0] is 1/sqrt(2).
        patch_a = np.zeros((2, 1, 1, 2), np.float32)
        patch_b = np.zeros((2, 1, 1, 2), np.float32)
        patch_a[:, 0, 0] = np.array([[1, 1], [1, 0]])
        patch_b[:, 0, 0] = np.array([[0, 1], [1, 1]])
        score = gaussian_fuse_features_before_similarity(
            [patch_a, patch_b],
            [(slice(0, 1), slice(0, 1), slice(0, 2)), (slice(0, 1), slice(0, 1), slice(1, 3))],
            np.ones((1, 1, 2), np.float32),
            (1, 1, 3),
            np.array([1, 0], np.float32),
        )
        self.assertAlmostEqual(float(score[0, 0, 1]), 1 / np.sqrt(2), places=6)
        self.assertNotAlmostEqual(float(score[0, 0, 1]), 0.5, places=6)

    def test_random_multiseed_summary(self):
        prototype = np.array([[1, 2, 3, 4]], np.float32)
        banks = [shuffled_prototype_bank(prototype, seed) for seed in range(42, 52)]
        self.assertEqual(len(banks), 10)
        self.assertTrue(all(np.allclose(np.linalg.norm(bank, axis=1), 1) for bank in banks))
        rows = []
        for index, seed in enumerate(range(42, 52)):
            rows.extend((
                {"random_seed": seed, "metric": float(index)},
                {"random_seed": seed, "metric": float(index + 2)},
            ))
        summary = random_seed_summary(rows, "metric")
        self.assertEqual(summary["n_seeds"], 10)
        self.assertEqual(summary["seed_case_means"]["42"], 1.0)
        self.assertEqual(summary["seed_case_means"]["51"], 10.0)
        self.assertAlmostEqual(summary["mean"], 5.5)
        self.assertGreater(summary["std"], 0)
        comparison = paired_comparison([2, 3, 4], [1, 1, 1], samples=500)
        self.assertEqual(comparison["n"], 3)
        self.assertEqual(len(comparison["case_differences"]), 3)

    def test_summary_reports_fixed_alpha_and_seed_level_controls(self):
        def localization(bank, value, seed=""):
            return {
                "case_id": "q", "bank": bank,
                "score_mode": "native" if bank == "baseline_probability" else "max",
                "random_seed": seed, "auprc": value,
                "dice_at_oracle_volume": value, "fn_region_auprc": value,
                "fn_recall_at_1v": value, "recall_at_1v": value,
                "recall_at_2v": value, "recall_at_5v": value,
            }

        rows = [
            localization("baseline_probability", 0.2),
            localization("oracle", 0.6),
            localization("prediction", 0.4),
        ]
        rows.extend(localization("random", 0.1 + seed / 10000, seed) for seed in range(42, 52))
        fusion = []
        for alpha in (0.25, 0.5, 1.0, 2.0):
            fusion.append({
                "case_id": "q", "bank": "oracle", "score_mode": "max", "alpha": alpha,
                "delta_dice": 0.03, "delta_recall": 0.2, "delta_precision": -0.05,
                "effective_empty_recovery": True, "baseline_empty_prediction": True,
                "dice": 0.1, "recall": 0.2, "precision": 0.1,
                "tp": 10, "fp": 90, "lesion_hits": 1,
            })
        result = summarize(rows, fusion, {}, [])
        self.assertEqual(result["random_control"]["auprc"]["n_seeds"], 10)
        self.assertAlmostEqual(result["case_level_mean_median"]["oracle"]["auprc"]["mean"], 0.6)
        self.assertEqual(result["answers"]["fixed_alpha_improved_declined_unchanged"]["1.0"]["improved"], 1)
        self.assertEqual(result["answers"]["empty_prediction_effective_recovery"]["1.0"]["recovered"], 1)
        self.assertFalse(result["answers"]["recall_precision_tradeoff"]["1.0"]["recall_gain_with_severe_precision_loss"])

    def test_one_tp_is_not_effective_empty_recovery(self):
        weak = {"dice": 0.001, "recall": 0.001, "tp": 1, "fp": 0}
        self.assertFalse(effective_empty_recovery(True, weak, 1))
        adequate = {"dice": 0.06, "recall": 0.11, "tp": 10, "fp": 50}
        self.assertTrue(effective_empty_recovery(True, adequate, 1))

    def test_return_features_false_preserves_decoder_output(self):
        from voxtell.model.voxtell_model import VoxTellDecoder, VoxTellModel

        self.assertIs(inspect.signature(VoxTellModel.forward).parameters["return_features"].default, False)
        encoder = SimpleNamespace(
            output_channels=[2] * 6, conv_op=torch.nn.Conv3d,
            strides=[1, 2, 2, 2, 2, 2], conv_bias=False,
            norm_op=None, norm_op_kwargs=None, dropout_op=None,
            dropout_op_kwargs=None, nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": False}, kernel_sizes=[3] * 6,
        )
        decoder = VoxTellDecoder(encoder, 1, [1] * 5, False, num_maskformer_stages=5, num_heads=1).eval()
        skips = [torch.randn(1, 2, size, size, size) for size in (32, 16, 8, 4, 2, 1)]
        embeddings = [torch.randn(1, 1, 2) for _ in range(5)]
        with torch.no_grad():
            omitted_default = decoder(skips, embeddings)
            original_interface = decoder(skips, embeddings, return_features=False)
            opted_in, feature = decoder(skips, embeddings, return_features=True)
        torch.testing.assert_close(omitted_default[0], original_interface[0])
        torch.testing.assert_close(original_interface[0], opted_in[0])
        self.assertEqual(tuple(feature.shape), (1, 2, 32, 32, 32))

    def test_predictor_interfaces_are_restored(self):
        from voxtell.inference.predictor import VoxTellPredictor
        from voxtell.inference.predictor_multiclass import VoxTellPredictor as MulticlassPredictor
        from voxtell.inference.oracle_support_full_volume import main

        self.assertEqual(
            inspect.signature(VoxTellPredictor.__init__).parameters["text_encoding_model"].default,
            "Qwen/Qwen3-Embedding-4B",
        )
        self.assertNotIn("batch_size", inspect.signature(VoxTellPredictor.embed_text_prompts).parameters)
        self.assertEqual(
            inspect.signature(MulticlassPredictor.__init__).parameters["text_encoding_model"].default,
            "Qwen/Qwen3-Embedding-4B",
        )
        self.assertIn("batch_size", inspect.signature(MulticlassPredictor.embed_text_prompts).parameters)
        main_source = inspect.getsource(main)
        self.assertLess(main_source.index("text = predictor.embed_text_prompts"), main_source.index("predictor.network = predictor.network.to"))


if __name__ == "__main__":
    unittest.main()
