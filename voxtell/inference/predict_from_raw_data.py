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
from typing import List, Optional, Tuple

import numpy as np
import torch

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from voxtell.inference.predictor import VoxTellPredictor
from voxtell.utils.image_augmentation import apply_contrast_enhancement, save_reoriented_nifti
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
        help='Save all prompts in a single multi-label file (WARNING: overlapping structures will be overwritten by later prompts)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--alignment',
        action='store_true',
        help='Compute prompt similarity and prompt-to-vision alignment metrics'
    )

    parser.add_argument(
        '--tsne',
        action='store_true',
        help='Save t-SNE plot of prompt and foreground vision embeddings (requires scikit-learn and matplotlib)'
    )

    parser.add_argument(
        '--tsne-output',
        type=str,
        default=None,
        help='Output path for the t-SNE plot (default: <output>/<case>_tsne.png)'
    )

    parser.add_argument(
        '--contrast-factor',
        type=float,
        default=1.0,
        help=(
            'Apply contrast enhancement before inference. '
            'Use 1.0 to keep the image unchanged. '
            'Augmented images are saved to the output folder when this value differs from 1.0.'
        )
    )

    return parser.parse_args()


def format_alignment_output(
    prompts: List[str],
    prompt_similarity: np.ndarray,
    prompt_vision_similarity: np.ndarray,
    foreground_voxels: List[int],
    tsne_path: Optional[str]
) -> str:
    lines = []
    lines.append("\nPrompt order:")
    lines.append("  " + ", ".join(prompts))
    lines.append("\nPrompt embedding cosine similarity matrix:")
    lines.append(np.array2string(prompt_similarity, precision=4, floatmode="fixed"))
    lines.append("\nForeground vision embedding vs prompt cosine similarity:")
    for prompt, similarity, voxels in zip(prompts, prompt_vision_similarity, foreground_voxels):
        similarity_str = "nan" if np.isnan(similarity) else f"{similarity:.4f}"
        lines.append(f"  {prompt}: {similarity_str} (foreground voxels: {voxels})")
    if tsne_path:
        lines.append(f"\nSaved t-SNE plot to: {tsne_path}")
    return "\n".join(lines)


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

    # Prepare output folder and filename metadata
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

    augmented_img = img
    if args.contrast_factor != 1.0:
        if args.verbose:
            print(f"Applying contrast enhancement (factor={args.contrast_factor})")
        augmented_img = apply_contrast_enhancement(img, args.contrast_factor)
        augmented_tag = f"contrast{args.contrast_factor:g}"
        augmented_path = output_folder / f"{input_filename}_{augmented_tag}{suffix}"
        save_reoriented_nifti(augmented_img, str(augmented_path), props)
        print(f"Saved augmented image to: {augmented_path}")

    if args.verbose:
        print(f"Image shape: {augmented_img.shape}")
        print(f"Text prompts: {args.prompts}")
        print(f"Loading VoxTell model from: {model_path}")

    predictor = VoxTellPredictor(
        model_dir=str(model_path),
        device=device
    )

    # Run prediction
    if args.verbose:
        print("Running prediction...")

    run_alignment = args.alignment or args.tsne
    alignment_output: Optional[Tuple[np.ndarray, np.ndarray, List[int], Optional[str]]] = None
    if run_alignment:
        tsne_output = None
        if args.tsne:
            tsne_output = args.tsne_output
            if tsne_output is None:
                tsne_output = str(output_folder / f"{input_filename}_tsne.png")
        segmentations, alignment = predictor.predict_single_image_with_alignment(
            augmented_img,
            args.prompts,
            tsne_output=tsne_output
        )
        alignment_output = (
            alignment["prompt_similarity"],
            alignment["prompt_vision_similarity"],
            alignment["foreground_voxels"],
            alignment["tsne_path"],
        )
    else:
        segmentations = predictor.predict_single_image(augmented_img, args.prompts)

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

    if alignment_output is not None:
        prompt_similarity, prompt_vision_similarity, foreground_voxels, tsne_path = alignment_output
        print(
            format_alignment_output(
                args.prompts,
                prompt_similarity,
                prompt_vision_similarity,
                foreground_voxels,
                tsne_path
            )
        )

    if args.verbose:
        print("\nPrediction completed successfully!")

    return 0


def predict_batch():
    prompts = ["spleen", "right_kidney", "left_kidney", "gallbladder", "liver", "stomach", "esophagus",
               "inferior_vena_cava", "pancreas", "duodenum"]
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

            segmentations = predictor.predict_single_image(img, prompts)  # ndarray:(P,Z,X,Y) {0，1}

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
