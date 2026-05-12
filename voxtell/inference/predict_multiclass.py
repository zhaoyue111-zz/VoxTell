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

from voxtell.inference.predictor_multiclass import VoxTellPredictor
from voxtell.utils.metrics_multiclass import dice_iou, compute_metrics

BINARY_THRESHOLD = 0.5


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
        segmentation: Segmentation or probability array to save.
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
        '--output-type',
        type=str,
        default='binary',
        choices=['binary', 'probabilities', 'logits'],
        help='Output type for predictions: binary, probabilities, or logits'
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

    if args.save_combined and args.output_type != 'binary':
        print(
            "Error: --save-combined requires --output-type binary because "
            "combined outputs need discrete masks, not probability or logit maps. "
            "Please use --output-type binary or remove --save-combined.",
            file=sys.stderr
        )
        return 1

    predictor = VoxTellPredictor(
        model_dir=str(model_path),
        device=device
    )

    # Run prediction
    if args.verbose:
        print("Running prediction...")

    segmentations = predictor.predict_single_image(
        img,
        args.prompts,
        output_type=args.output_type
    )

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
    prompts = ["spleen", "right_kidney", "left_kidney", "gallbladder", "liver", "stomach", "esophagus",
               "inferior_vena_cava", "pancreas", "duodenum", "aorta"]
    print("\nLabel mapping:")
    for i, prompt in enumerate(prompts):  # 如果要combined segmentations，需要标签和和提示引引齐齐
        print(f"  {i + 1}: {prompt}")

    device = torch.device(f'cpu')
    model_path = Path("D:\\pythonCode\\VoxTell\\model")
    predictor = VoxTellPredictor(model_dir=str(model_path), device=device)

    output_folder = Path("./out_multi/Delay")
    output_folder.mkdir(parents=True, exist_ok=True)

    input_path = Path("D:\\dataset\\CT_MRI_DATA_3D\\images\\Delay")
    mask_path = Path("D:\\dataset\\CT_MRI_DATA_3D\\labels\\Delay")
    num = sum(1 for f in os.listdir(input_path) if f.endswith(".nii.gz"))

    num_classes=len(prompts)
    total_class_dices = np.zeros(num_classes)
    total_class_ious = np.zeros(num_classes)
    for filename in os.listdir(input_path):
        if filename.endswith('.nii.gz'):
            image_path = os.path.join(input_path, filename)

            reader_writer = get_reader_writer(str(image_path))
            img, props = reader_writer.read_images([str(image_path)])  # img:ndarray(P,Z,X,Y) [-1,1]

            probabilities = predictor.predict_single_image(
                img,
                prompts,
                output_type="probabilities"
            )  # ndarray:(P,Z,X,Y) [0,1]
            segmentations = (probabilities > BINARY_THRESHOLD).astype(np.uint8)

            for i, prompt in enumerate(prompts):
                save_segmentation(
                    probabilities[i],
                    output_folder,
                    filename,
                    props,
                    prompt_name=f"prob_map--{prompt}",
                    suffix="nii.gz"
                )

            combined_seg = np.zeros_like(segmentations[0], dtype=np.uint8) # [Z,X,Y]
            for i, seg in enumerate(segmentations):
                combined_seg[seg > 0] = i + 1 # [1,Z,X,Y]
            save_segmentation(combined_seg, output_folder, filename, props, suffix="nii.gz")

            gt_path = os.path.join(mask_path, filename)
            gt, _ = reader_writer.read_images([str(gt_path)])  # ndarray:(P,Z,X,Y)

            # dice,iou=dice_iou(segmentations,gt)
            # combined_seg=np.expand_dims(combined_seg,axis=0) # [1,Z,X,Y]
            gt_3d = gt[0]
            num_classes = len(prompts)
            gt_one_hot = np.zeros((num_classes, *gt_3d.shape), dtype=np.uint8)  # [P,Z,X,Y]
            for i in range(num_classes):
                gt_one_hot[i][gt_3d == (i + 1)] = 1
            dice, iou = compute_metrics(segmentations, gt_one_hot)
            print(f"\nResults for {filename}:")
            for i, name in enumerate(prompts):
                print(f"  {name:20s}: Dice {dice[i]:.4f}, IoU {iou[i]:.4f}")

            total_class_dices += np.array(dice)
            total_class_ious += np.array(iou)

    mean_class_dices = total_class_dices / num
    mean_class_ious = total_class_ious / num

    print("\n" + "=" * 40)
    print(f"{'Class Name':20s} | {'Mean Dice':10s} | {'Mean IoU':10s}")
    print("-" * 40)
    for i, name in enumerate(prompts):
        print(f"{name:20s} | {mean_class_dices[i]:.4f}     | {mean_class_ious[i]:.4f}")

    print("-" * 40)
    print(f"{'OVERALL AVERAGE':20s} | {np.mean(mean_class_dices):.4f}     | {np.mean(mean_class_ious):.4f}")
    print("=" * 40)

    return 0


if __name__ == '__main__':
    os.environ['HF_HUB_OFFLINE'] = '1'
    predict_batch()
