#!/usr/bin/env python3
"""
Command-line entrypoint for VoxTell segmentation prediction.

This script provides a CLI interface to run VoxTell predictions on medical images
with free-text prompts.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from voxtell.inference.predictor import VoxTellPredictor
from voxtell.utils.metrics import dice_iou, compute_metrics, compute_boundary_metrics


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

  # Save combined multi-label file using argmax across prompts
  voxtell-predict -i case001.nii.gz -o output_folder -m /path/to/model -p "liver" "spleen" --save-combined --combine-strategy argmax --combine-threshold 0.5

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
        required=True,
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
        default='cuda',
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
        help='Save all prompts in a single multi-label file (use --combine-strategy to control overlap handling)'
    )

    parser.add_argument(
        '--combine-strategy',
        type=str,
        default='overwrite',
        choices=['overwrite', 'argmax'],
        help='Combine strategy for --save-combined: overwrite (later prompts overwrite earlier ones) or argmax (highest probability per voxel)'
    )

    parser.add_argument(
        '--combine-threshold',
        type=float,
        default=0.5,
        help='Background threshold for argmax combine strategy (ignored for overwrite); voxels with max probability below this are set to 0'
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

    if not 0.0 <= args.combine_threshold <= 1.0:
        raise ValueError("--combine-threshold must be between 0 and 1")

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    if not (model_path / 'plans.json').exists():
        raise FileNotFoundError(f"plans.json not found in model directory: {model_path}")

    if not (model_path / 'fold_0' / 'checkpoint_final.pth').exists():
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

    output_type = (
        "probabilities"
        if args.save_combined and args.combine_strategy == "argmax"
        else "binary"
    )
    segmentations = predictor.predict_single_image(img, args.prompts, output_type=output_type)

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
        if args.combine_strategy == "argmax":
            if len(args.prompts) == 1:
                combined_seg = (segmentations[0] >= args.combine_threshold).astype(np.uint8)
            else:
                max_probs = np.max(segmentations, axis=0)
                combined_seg = np.argmax(segmentations, axis=0).astype(np.uint8) + 1
                combined_seg[max_probs < args.combine_threshold] = 0

            save_segmentation(combined_seg, output_folder, input_filename, props, suffix=suffix)
            print(f"\nArgmax combine threshold: {args.combine_threshold}")
            if len(args.prompts) > 1:
                print("\nLabel mapping:")
                for i, prompt in enumerate(args.prompts):
                    print(f"  {i + 1}: {prompt}")
        else:
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
    prompts = ["spleen", "right_kidney", "left_kidney", "gallbladder", "liver", "stomach", "esophagus",
               "inferior_vena_cava", "pancreas", "duodenum"]
    combine_strategy = "argmax"
    combine_threshold = 0.5
    print("\nLabel mapping:")
    for i, prompt in enumerate(prompts):
        print(f"  {i + 1}: {prompt}")

    device = torch.device(f'cuda:0')
    model_path = Path("/home/data4/zy/weight/voxtell")
    predictor = VoxTellPredictor(model_dir=str(model_path), device=device)

    output_folder = Path("./out/Delay_multi")
    output_folder.mkdir(parents=True, exist_ok=True)

    input_path = Path("/home/data4/zy/data/CT_MRI_DATA/images/Delay")
    mask_path = Path("/home/data4/zy/data/CT_MRI_DATA/labels/Delay")
    num = sum(1 for f in os.listdir(input_path) if f.endswith(".nii.gz"))

    dices = 0.0
    ious = 0.0
    hds = 0.0
    asds = 0.0
    for filename in os.listdir(input_path):
        if filename.endswith('.nii.gz'):
            image_path = os.path.join(input_path, filename)

            reader_writer = get_reader_writer(str(image_path))
            img, props = reader_writer.read_images([str(image_path)])  # img:ndarray(P,Z,X,Y) [-1,1]

            output_type = "probabilities" if combine_strategy == "argmax" else "binary"
            segmentations = predictor.predict_single_image(
                img, prompts, output_type=output_type
            )  # ndarray:(P,Z,X,Y)

            if combine_strategy == "argmax":
                max_probs = np.max(segmentations, axis=0)
                combined_seg = np.argmax(segmentations, axis=0).astype(np.uint8) + 1
                combined_seg[max_probs < combine_threshold] = 0
            else:
                combined_seg = np.zeros_like(segmentations[0], dtype=np.uint8)
                for i, seg in enumerate(segmentations):
                    combined_seg[seg > 0] = i + 1
            save_segmentation(combined_seg, output_folder, filename, props, suffix="nii.gz")

            gt_path = os.path.join(mask_path, filename)
            gt, _ = reader_writer.read_images([str(gt_path)])  # ndarray:(P,Z,X,Y)

            break
            # dice,iou=dice_iou(segmentations,gt)
    #         dice, iou = compute_metrics(segmentations, gt)
    #         hd, asd_val = compute_boundary_metrics(segmentations, gt, spacing=props['spacing'])
    #         print(f"{filename} dice: {dice:.4f}, iou: {iou:.4f}, hd:{hd:.4f}, asd_val:{asd_val:.4f}")
    #         dices += dice
    #         ious += iou
    #         hds += hd
    #         asds += asd_val
    #
    # mdice = dices * 1.0 / num
    # miou = ious * 1.0 / num
    # mhd = hds * 1.0 / num
    # masd = asds * 1.0 / num
    # print(f"\ndice: {mdice:.4f}, miou: {miou:.4f}, mhd: {mhd:.4f}, masd: {masd:.4f}")

    return 0


if __name__ == '__main__':
    os.environ['HF_HUB_OFFLINE'] = '1'
    predict_batch()
