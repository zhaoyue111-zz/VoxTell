#!/usr/bin/env python3
"""
Command-line entrypoint for VoxTell segmentation prediction.

This script provides a CLI interface to run VoxTell predictions on medical images
with free-text prompts.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

# These must be set before importing predictor_multiclass/transformers.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from voxtell.inference.predictor_multiclass import VoxTellPredictor
from voxtell.utils.metrics_multiclass import compute_metrics_from_label_map


PREDICTION_THRESHOLD = 0.5
OVERALL_HISTOGRAM_BINS = 500_000


def case_background_probability_stats(
        foreground_probability: np.ndarray,
        gt_foreground: np.ndarray,
        predicted_foreground: Optional[np.ndarray] = None,
        threshold: float = PREDICTION_THRESHOLD,
) -> dict:
    """Summarize foreground probabilities among predicted-background pixels."""
    probability = np.asarray(foreground_probability, dtype=np.float32)
    gt_foreground = np.asarray(gt_foreground, dtype=bool)
    if probability.shape != gt_foreground.shape:
        raise ValueError(
            f"Probability/GT shape mismatch: {probability.shape} vs {gt_foreground.shape}"
        )

    if predicted_foreground is None:
        # The predictor's existing binary path thresholds sigmoid probabilities with > 0.5.
        # Preserve that exact decision rule, including the treatment of values equal to 0.5.
        predicted_background = probability <= threshold
    else:
        predicted_foreground = np.asarray(predicted_foreground, dtype=bool)
        if predicted_foreground.shape != probability.shape:
            raise ValueError(
                "Prediction/Probability shape mismatch: "
                f"{predicted_foreground.shape} vs {probability.shape}"
            )
        predicted_background = ~predicted_foreground
    tn_values = probability[predicted_background & ~gt_foreground]
    fn_values = probability[predicted_background & gt_foreground]

    def summarize(values: np.ndarray, prefix: str, quantiles: dict) -> dict:
        result = {f"{prefix}_count": int(values.size)}
        if values.size == 0:
            result.update({key: float("nan") for key in quantiles})
            return result
        for key, statistic in quantiles.items():
            if statistic == "mean":
                result[key] = float(np.mean(values, dtype=np.float64))
            elif statistic == "min":
                result[key] = float(np.min(values))
            elif statistic == "max":
                result[key] = float(np.max(values))
            else:
                result[key] = float(np.percentile(values, statistic))
        return result

    stats = summarize(
        tn_values,
        "tn",
        {
            "tn_fg_prob_mean": "mean",
            "tn_fg_prob_max": "max",
            "tn_fg_prob_p95": 95,
        },
    )
    stats.update(summarize(
        fn_values,
        "fn",
        {
            "fn_fg_prob_min": "min",
            "fn_fg_prob_mean": "mean",
            "fn_fg_prob_max": "max",
            "fn_fg_prob_p90": 90,
            "fn_fg_prob_p95": 95,
        },
    ))
    return stats


class StreamingProbabilityStats:
    """Pool pixel statistics without retaining all test-volume probabilities."""

    def __init__(self, bins: int = OVERALL_HISTOGRAM_BINS):
        self.bins = int(bins)
        self.count = 0
        self.total = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.histogram = np.zeros(self.bins, dtype=np.int64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return
        self.count += int(values.size)
        self.total += float(np.sum(values, dtype=np.float64))
        self.minimum = min(self.minimum, float(np.min(values)))
        self.maximum = max(self.maximum, float(np.max(values)))
        bin_indices = np.floor(
            values * (self.bins / PREDICTION_THRESHOLD)
        ).astype(np.int64)
        np.clip(bin_indices, 0, self.bins - 1, out=bin_indices)
        self.histogram += np.bincount(bin_indices, minlength=self.bins)

    def percentile(self, q: float) -> float:
        if self.count == 0:
            return float("nan")
        rank = (self.count - 1) * (q / 100.0)
        lower_rank = int(np.floor(rank))
        upper_rank = int(np.ceil(rank))
        cumulative = np.cumsum(self.histogram)

        def value_at_rank(item_rank: int) -> float:
            bin_index = int(np.searchsorted(cumulative, item_rank + 1, side="left"))
            # Return the bin center; the worst-case quantile error is half a bin.
            return (bin_index + 0.5) * (PREDICTION_THRESHOLD / self.bins)

        lower = value_at_rank(lower_rank)
        upper = value_at_rank(upper_rank)
        return float(lower + (rank - lower_rank) * (upper - lower))

    def summary(self, prefix: str, requested: tuple[str, ...]) -> dict:
        result = {f"{prefix}_count": self.count}
        for statistic in requested:
            key = f"{prefix}_fg_prob_{statistic}"
            if self.count == 0:
                result[key] = float("nan")
            elif statistic == "mean":
                result[key] = self.total / self.count
            elif statistic == "min":
                result[key] = self.minimum
            elif statistic == "max":
                result[key] = self.maximum
            else:
                result[key] = self.percentile(float(statistic[1:]))
        return result


def get_reader_writer(file_path: str):
    """
    Determine the appropriate reader/writer based on file extension.

    Args:
        file_path: Path to the input file.

    Returns:
        Appropriate reader/writer instance.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix in ['.nii', '.gz']:
        return NibabelIOWithReorient()
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Only NIfTI format (.nii, .nii.gz) is currently supported. "
            "Images must be reorientable to standard orientation with correct metadata."
        )


def save_segmentation(
        segmentation: np.ndarray,
        output_folder: Path,
        input_filename: str,
        properties: dict,
        prompt_name: str = None,
        suffix: str = '.nii.gz'
) -> None:
    """
    Save segmentation mask to file.

    Args:
        segmentation: Segmentation array to save.
        output_folder: Output folder path.
        input_filename: Original input filename (without extension).
        properties: Image properties from the reader.
        prompt_name: Optional prompt name to include in filename.
        suffix: File extension to use.
    """
    if prompt_name:
        # Clean prompt name for filename
        safe_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in prompt_name)
        safe_name = safe_name.replace(' ', '_')
        output_file = output_folder / f"{input_filename}_{safe_name}"
    else:
        output_file = output_folder / f"{input_filename}"

    # Use NIfTI writer
    reader_writer = NibabelIOWithReorient()
    reader_writer.write_seg(segmentation, str(output_file), properties)
    print(f"Saved segmentation to: {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt (saves to output_folder/case001_liver.nii.gz)
  voxtell-predict -i case001.nii.gz -o output_folder -m /path/to/model -p "liver"

  # Multiple prompts (saves individual files by default)
  voxtell-predict -i case001.nii.gz -o output_folder -m /path/to/model -p "liver" "spleen" "kidney"

  # Save combined multi-label file (with overlap warning)
  voxtell-predict -i case001.nii.gz -o output_folder -m /path/to/model -p "liver" "spleen" --save-combined

  # Use CPU
  voxtell-predict -i case001.nii.gz -o output_folder -m /path/to/model -p "liver" --device cpu
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input image file (NIfTI format recommended)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Path to output folder where segmentation files will be saved'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        default="D:\\pythonCode\\VoxTell\\model",
        help='Path to VoxTell model directory containing plans.json and fold_0/'
    )

    parser.add_argument(
        '-p', '--prompts',
        type=str,
        nargs='+',
        required=True,
        help='Text prompt(s) for segmentation (e.g., "liver" "spleen" "tumor")'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cuda', 'cpu'],
        help='Device to use for inference (default: cuda)'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID to use (default: 0)'
    )

    parser.add_argument(
        '--save-combined',
        action='store_true',
        help='Save all prompts in a single multi-label file (WARNING: overlapping structures will be overwritten by later prompts)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def main() -> int:
    """Main entrypoint function."""
    args = parse_args()

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    if not (model_path / 'plans.json').exists():
        raise FileNotFoundError(f"plans.json not found in model directory: {model_path}")

    if not (model_path / 'checkpoint_final.pth').exists():
        raise FileNotFoundError(f"checkpoint_final.pth not found in {model_path / 'fold_0'}")

    # Setup device
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU", file=sys.stderr)
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{args.gpu}')
            if args.verbose:
                print(f"Using GPU: {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        device = torch.device('cpu')
        if args.verbose:
            print("Using CPU")

    # Load image
    if args.verbose:
        print(f"Loading image: {input_path}")

    try:
        reader_writer = get_reader_writer(str(input_path))
        img, props = reader_writer.read_images([str(input_path)])
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Image shape: {img.shape}")
        print(f"Text prompts: {args.prompts}")
        print(f"Loading VoxTell model from: {model_path}")

    predictor = VoxTellPredictor(
        model_dir=str(model_path),
        device=device
    )

    # Run prediction
    if args.verbose:
        print("Running prediction...")

    segmentations = predictor.predict_single_image(img, args.prompts)

    # Save results
    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get input filename without extension
    input_filename = input_path.stem
    if input_filename.endswith('.nii'):
        input_filename = input_filename[:-4]

    # Determine file suffix from input
    if input_path.suffix == '.gz' and input_path.stem.endswith('.nii'):
        suffix = '.nii.gz'
    else:
        suffix = input_path.suffix

    if args.save_combined:
        # Show warning about overlapping structures
        if len(args.prompts) > 1:
            print("\n" + "=" * 80)
            print("WARNING: Saving combined multi-label segmentation.")
            print("If prompts generate overlapping structures, later prompts will overwrite")
            print("earlier ones. This may result in loss of segmentation information.")
            print("Consider using individual file output (default) for overlapping structures.")
            print("=" * 80 + "\n")

        # Save all prompts in a single multi-label file
        if len(args.prompts) == 1:
            # Single prompt - save as-is
            save_segmentation(segmentations[0], output_folder, input_filename, props, suffix=suffix)
        else:
            # Multiple prompts - create multi-label segmentation
            # Each prompt gets a different label value (1, 2, 3, ...)
            # Later prompts overwrite earlier ones in case of overlap
            combined_seg = np.zeros_like(segmentations[0], dtype=np.uint8)
            for i, seg in enumerate(segmentations):
                combined_seg[seg > 0] = i + 1
            save_segmentation(combined_seg, output_folder, input_filename, props, suffix=suffix)

            print("\nLabel mapping:")
            for i, prompt in enumerate(args.prompts):
                print(f"  {i + 1}: {prompt}")
    else:
        # Default: Save each prompt as a separate file
        for i, prompt in enumerate(args.prompts):
            save_segmentation(
                segmentations[i],
                output_folder,
                input_filename,
                props,
                prompt_name=prompt,
                suffix=suffix
            )

    if args.verbose:
        print("\nPrediction completed successfully!")

    return 0


def predict_batch():
    prompts = ["spleen", "right_kidney", "left_kidney", "gallbladder", "liver", "stomach", "aorta",
               "inferior_vena_cava", "duodenum", "pancreas", "esophagus"]
    print("\nLabel mapping:")
    for i, prompt in enumerate(prompts):  # 如果要combined segmentations，需要标签和和提示引引齐齐
        print(f"  {i + 1}: {prompt}")

    device = torch.device(f'cuda')
    model_path = Path("/data/zy/VoxTell_from_disk/model")
    predictor = VoxTellPredictor(model_dir=str(model_path), device=device)
    # All four sequences use the same fixed prompts. Encode once in small chunks
    # to avoid keeping the text backbone and segmentation network on the GPU at once.
    text_embeddings = predictor.embed_text_prompts(prompts, batch_size=1)
    # Keep the full-volume Gaussian-fusion accumulator on host memory; patch
    # inference itself still runs on CUDA and preserves the same accumulation order.
    predictor.perform_everything_on_device = False

    sequences=["P1","PreArtery","PV","T2"]
    output_root = Path("./out_multi")
    per_case_rows = []
    all_tn_stats = StreamingProbabilityStats()
    all_fn_stats = StreamingProbabilityStats()
    stats_fieldnames = [
        "tn_count",
        "tn_fg_prob_mean",
        "tn_fg_prob_max",
        "tn_fg_prob_p95",
        "fn_count",
        "fn_fg_prob_min",
        "fn_fg_prob_mean",
        "fn_fg_prob_max",
        "fn_fg_prob_p90",
        "fn_fg_prob_p95",
    ]

    for s in sequences:
        output_folder = output_root / s
        output_folder.mkdir(parents=True, exist_ok=True)

        input_path = Path("/data/zy/CT_MRI_DATA_3D/images/"+s)
        mask_path = Path("/data/zy/CT_MRI_DATA_3D/labels/"+s)

        num_classes = len(prompts)
        total_class_dices = np.zeros(num_classes)
        total_class_ious = np.zeros(num_classes)
        processed_cases = 0
        filenames = sorted(f for f in os.listdir(input_path) if f.endswith(".nii.gz"))
        for filename in filenames:
            image_path = os.path.join(input_path, filename)

            reader_writer = get_reader_writer(str(image_path))
            img, props = reader_writer.read_images([str(image_path)])  # ndarray:(P,Z,Y,X)
            segmentations, foreground_probabilities = predictor.predict_single_image(
                img,
                prompts,
                return_probabilities=True,
                text_embeddings=text_embeddings,
            )

            combined_seg = np.zeros_like(segmentations[0], dtype=np.uint8)
            for class_index, segmentation in enumerate(segmentations):
                combined_seg[segmentation > 0] = class_index + 1
            save_segmentation(
                combined_seg,
                output_folder,
                filename,
                props,
                suffix=".nii.gz",
            )

            gt_path = os.path.join(mask_path, filename)
            gt, _ = reader_writer.read_images([str(gt_path)])
            dice, iou = compute_metrics_from_label_map(segmentations, gt[0])

            case_row = {"sequence": s, "case": filename}
            for class_index, prompt in enumerate(prompts):
                gt_foreground = gt[0] == (class_index + 1)
                probability = foreground_probabilities[class_index]
                case_stats = case_background_probability_stats(
                    probability,
                    gt_foreground,
                    predicted_foreground=segmentations[class_index],
                )
                for field, value in case_stats.items():
                    case_row[f"{prompt}_{field}"] = value

                predicted_background = ~segmentations[class_index].astype(bool)
                all_tn_stats.update(
                    probability[predicted_background & ~gt_foreground]
                )
                all_fn_stats.update(
                    probability[predicted_background & gt_foreground]
                )

            per_case_rows.append(case_row)

            print(f"\nResults for {filename}:")
            for i, name in enumerate(prompts):
                print(f"  {name:20s}: Dice {dice[i]:.4f}, IoU {iou[i]:.4f}")

            total_class_dices += np.asarray(dice)
            total_class_ious += np.asarray(iou)
            processed_cases += 1
            del img, segmentations, combined_seg, gt
            del foreground_probabilities

        if processed_cases == 0:
            raise RuntimeError(f"No .nii.gz images found in {input_path}")

        mean_class_dices = total_class_dices / processed_cases
        mean_class_ious = total_class_ious / processed_cases

        print(s+"\n")
        print("\n" + "=" * 40)
        print(f"{'Class Name':20s} | {'Mean Dice':10s} | {'Mean IoU':10s}")
        print("-" * 40)
        for i, name in enumerate(prompts):
            print(f"{name:20s} | {mean_class_dices[i]:.4f}     | {mean_class_ious[i]:.4f}")

        print("-" * 40)
        print(f"{'OVERALL AVERAGE':20s} | {np.mean(mean_class_dices):.4f}     | {np.mean(mean_class_ious):.4f}")
        print("=" * 40)

    output_root.mkdir(parents=True, exist_ok=True)
    probability_stats_csv = output_root / "tn_fn_foreground_probability.csv"
    fieldnames = ["sequence", "case"] + [
        f"{prompt}_{field}"
        for prompt in prompts
        for field in stats_fieldnames
    ]
    with probability_stats_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_case_rows)
    print(f"\nSaved per-case TN/FN foreground probability statistics to: {probability_stats_csv}")

    overall_tn = all_tn_stats.summary("tn", ("mean", "max", "p95"))
    overall_fn = all_fn_stats.summary("fn", ("min", "mean", "max", "p90", "p95"))
    print("\nOverall TN/FN foreground probability statistics (all sequences and prompts):")
    print(
        "TN: "
        f"count={overall_tn['tn_count']}, "
        f"mean={overall_tn['tn_fg_prob_mean']:.8g}, "
        f"max={overall_tn['tn_fg_prob_max']:.8g}, "
        f"p95={overall_tn['tn_fg_prob_p95']:.8g}"
    )
    print(
        "FN: "
        f"count={overall_fn['fn_count']}, "
        f"min={overall_fn['fn_fg_prob_min']:.8g}, "
        f"mean={overall_fn['fn_fg_prob_mean']:.8g}, "
        f"max={overall_fn['fn_fg_prob_max']:.8g}, "
        f"p90={overall_fn['fn_fg_prob_p90']:.8g}, "
        f"p95={overall_fn['fn_fg_prob_p95']:.8g}"
    )
    print(
        "Overall percentile estimates use a 1e-6 probability histogram "
        "(percentile granularity is approximately 1e-6)."
    )
    del text_embeddings

    return 0


if __name__ == '__main__':
    predict_batch()
