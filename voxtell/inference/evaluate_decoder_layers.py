#!/usr/bin/env python3
"""Evaluate Dice/mIoU for every image-decoder deep-supervision head."""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from voxtell.inference.predictor_multiclass import VoxTellPredictor
from voxtell.utils.metrics_multiclass import compute_metrics


DEFAULT_PROMPTS = ["target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Dice and mIoU for all VoxTell image-decoder outputs."
    )
    parser.add_argument(
        "--images", required=True,
        help="Image directory (or a *.nii.gz glob), for example xxx/images/P0.",
    )
    parser.add_argument("--labels", required=True, help="Directory containing label NIfTI files.")
    parser.add_argument(
        "--model", default=r"D:\pythonCode\VoxTell_from_disk\model",
        help="Model directory containing plans.json and checkpoint_final.pth.",
    )
    parser.add_argument(
        "--text-model", default="Qwen/Qwen3-Embedding-4B",
        help="Local Qwen embedding model directory, or Hugging Face model id. "
             "Example: /mnt/afs2/models/Qwen3-Embedding-4B.",
    )
    parser.add_argument(
        "--prompts", nargs="+", default=DEFAULT_PROMPTS,
        help="Text prompts in label order. For one prompt, all non-zero GT labels are used.",
    )
    parser.add_argument(
        "--label-values", type=int, nargs="+", default=None,
        help="Optional GT label values in prompt order. For example: --label-values 5 "
             "or --label-values 1 2 3. If omitted, one prompt uses all non-zero labels; "
             "multiple prompts use sorted non-zero values.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N cases.")
    parser.add_argument(
        "--csv", default=None,
        help="Optional CSV output path for per-case/per-layer metrics.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for the first case's decoder predictions and GT. "
             "Defaults to the CSV parent directory, or ./output.",
    )
    return parser.parse_args()


def find_cases(images_path: Path, labels_dir: Path) -> List[Tuple[str, Path, Path]]:
    cases = []
    if images_path.is_dir():
        image_paths = sorted(images_path.glob("*.nii.gz"))
    else:
        image_paths = sorted(images_path.parent.glob(images_path.name))

    for image_path in image_paths:
        label = labels_dir / image_path.name
        if label.is_file():
            case_id = image_path.name
            if case_id.endswith(".nii.gz"):
                case_id = case_id[:-len(".nii.gz")]
            else:
                case_id = image_path.stem
            cases.append((case_id, image_path, label))
    return cases


def main() -> int:
    args = parse_args()
    images_path = Path(args.images)
    labels_dir = Path(args.labels)
    if not images_path.exists() and not images_path.parent.is_dir():
        raise FileNotFoundError(images_path)
    if not labels_dir.is_dir():
        raise FileNotFoundError(labels_dir)

    cases = find_cases(images_path, labels_dir)
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise RuntimeError("No complete 4-channel image/label cases were found.")

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        if args.device == "cuda":
            print("CUDA is unavailable; using CPU.")
        device = torch.device("cpu")

    if args.label_values is not None and len(args.label_values) != len(args.prompts):
        raise ValueError("--label-values must contain one value per prompt.")

    predictor = VoxTellPredictor(
        model_dir=str(args.model),
        device=device,
        text_encoding_model=args.text_model,
    )
    print("Decoder mapping (actual upsampling order; model list is high->low):")
    for info in predictor.decoder_output_metadata:
        print("  " + json.dumps(info, sort_keys=True))
    reader = NibabelIOWithReorient()
    output_dir = Path(args.output) if args.output else (
        Path(args.csv).parent if args.csv else Path("output")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # The predictor returns highest->lowest resolution.  Use decoder metadata
    # instead of treating list position as a stage number.
    decoder_metadata = predictor.decoder_output_metadata
    n_layers = len(decoder_metadata)
    dice_sum = np.zeros((n_layers, len(args.prompts)), dtype=np.float64)
    iou_sum = np.zeros_like(dice_sum)
    rows = []

    print(f"Evaluating {len(cases)} cases on {device}...")
    for case_idx, (case_id, image_path, label_path) in enumerate(cases, start=1):
        image, image_props = reader.read_images([str(image_path)])
        label, _ = reader.read_images([str(label_path)])
        gt_labels = np.rint(label[0]).astype(np.int64)

        nonzero_values = sorted(int(v) for v in np.unique(gt_labels) if v != 0)
        if args.label_values is not None:
            label_values = args.label_values
        elif len(args.prompts) == 1:
            label_values = [None]
        else:
            if len(nonzero_values) < len(args.prompts):
                raise ValueError(
                    f"{case_id}: found GT labels {nonzero_values}, but received "
                    f"{len(args.prompts)} prompts. Pass --label-values explicitly."
                )
            label_values = nonzero_values[:len(args.prompts)]

        if label_values[0] is None:
            gt = (gt_labels != 0)[None].astype(np.uint8)
        else:
            gt = np.stack([(gt_labels == value).astype(np.uint8)
                           for value in label_values], axis=0)

        layer_predictions = predictor.predict_single_image(
            image, args.prompts, output_type="binary", return_all_layers=True
        )

        # Save only the first case to keep the output compact. Predictions are
        # written as a binary mask for one prompt, or as a combined label map
        # (1..N) for multiple prompts. The GT keeps its original label values.
        if case_idx > 0:
            case_output_dir = output_dir / case_id
            case_output_dir.mkdir(parents=True, exist_ok=True)

            for layer_info in decoder_metadata:
                layer_idx = int(layer_info["decoder_stage"])
                prediction = layer_predictions[int(layer_info["model_output_list_index"])]
                if prediction.shape[0] == 1:
                    prediction_to_save = prediction[0].astype(np.uint8)
                else:
                    prediction_to_save = np.zeros(
                        prediction.shape[1:], dtype=np.uint8
                    )
                    for prompt_idx in range(prediction.shape[0]):
                        prediction_to_save[prediction[prompt_idx] > 0] = prompt_idx + 1

                reader.write_seg(
                    prediction_to_save,
                    str(case_output_dir / f"predict_decoder_{layer_idx}.nii.gz"),
                    image_props,
                )

            reader.write_seg(
                gt_labels.astype(np.uint8),
                str(case_output_dir / "gt.nii.gz"),
                image_props,
            )
            print(f"Saved first-case predictions and GT to {case_output_dir}")

        print(f"[{case_idx}/{len(cases)}] {case_id}")
        for layer_info in decoder_metadata:
            layer_idx = int(layer_info["decoder_stage"])
            prediction = layer_predictions[int(layer_info["model_output_list_index"])]
            dice, iou = compute_metrics(prediction, gt)
            dice = np.asarray(dice)
            iou = np.asarray(iou)
            dice_sum[layer_idx - 1] += dice
            iou_sum[layer_idx - 1] += iou
            print(
                f"  decoder_{layer_idx}: Dice={dice.mean():.4f}, "
                f"mIoU={iou.mean():.4f}"
            )
            for class_idx, prompt in enumerate(args.prompts):
                rows.append({
                    "case": case_id,
                    "decoder_layer": layer_idx,
                    "prompt": prompt,
                    "dice": float(dice[class_idx]),
                    "miou": float(iou[class_idx]),
                })

    mean_dice = dice_sum / len(cases)
    mean_iou = iou_sum / len(cases)
    print("\nMean over cases")
    print("layer | Dice | mIoU")
    print("------|------|------")
    for layer_idx in range(n_layers):
        print(
            f"{layer_idx + 1:5d} | {mean_dice[layer_idx].mean():.4f} | "
            f"{mean_iou[layer_idx].mean():.4f}"
        )
    print("\nPer-class means")
    for layer_idx in range(n_layers):
        values = ", ".join(
            f"{prompt}: Dice={mean_dice[layer_idx, class_idx]:.4f}, "
            f"mIoU={mean_iou[layer_idx, class_idx]:.4f}"
            for class_idx, prompt in enumerate(args.prompts)
        )
        print(f"decoder_{layer_idx + 1}: {values}")

    if args.csv:
        import csv
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=("case", "decoder_layer", "prompt", "dice", "miou"))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved metrics to {csv_path}")
    return 0


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    raise SystemExit(main())

'''
python -m voxtell.inference.evaluate_decoder_layers \
  --images /mnt/afs2/zy/CT_MRI_DATA_3D/images/P0 \
  --labels /mnt/afs2/zy/CT_MRI_DATA_3D/labels/P0 \
  --model /mnt/afs2/zy/VoxTell_from_disk/model \
  --text-model /mnt/afs2/models/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b \
  --prompts liver \
  --device cuda \
  --output /mnt/afs2/zy/VoxTell_from_disk/output \
  --csv /mnt/afs2/zy/VoxTell_from_disk/output/decoder_metrics.csv
'''
