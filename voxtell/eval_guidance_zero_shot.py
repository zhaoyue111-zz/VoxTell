#!/usr/bin/env python3
from __future__ import annotations

'''
逐个评估 88 个 privacy_guidence.json prompt 的
  zero-shot 分割能力。
'''

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from voxtell.inference.predictor_multiclass import VoxTellPredictor


DEFAULT_CLASSES = [
    "spleen",
    "right_kidney",
    "left_kidney",
    "gallbladder",
    "liver",
    "stomach",
    "aorta",
    "inferior_vena_cava",
    "duodenum",
    "pancreas",
    "esophagus",
]

DEFAULT_CLASSES_TO_CONCEPTS = {
    0: list(range(0, 8)),
    1: list(range(8, 16)),
    2: list(range(16, 24)),
    3: list(range(24, 32)),
    4: list(range(32, 40)),
    5: list(range(40, 48)),
    6: list(range(48, 56)),
    7: list(range(56, 64)),
    8: list(range(64, 72)),
    9: list(range(72, 80)),
    10: list(range(80, 88)),
}


class TeeStream:
    def __init__(self, stream, log_file) -> None:
        self.stream = stream
        self.log_file = log_file

    def write(self, message: str) -> None:
        self.stream.write(message)
        self.log_file.write(message)

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every guidance prompt with original VoxTell zero-shot inference."
    )
    parser.add_argument("--data-root", default="/data/zy/CT_MRI_DATA_3D")
    parser.add_argument("--sequence", default="EAP")
    parser.add_argument("--model-dir", default="/data/zy/VoxTell_from_disk/model")
    parser.add_argument("--concepts-json", default="/data/zy/CT_MRI_DATA_3D/privacy_guidence.json")
    parser.add_argument("--output-dir", default="/data/zy/VoxTell_from_disk/guidance_zero_shot/EAP")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument(
        "--prompt-chunk-size",
        type=int,
        default=1,
        help="Number of guidance prompts evaluated per VoxTell forward batch.",
    )
    return parser.parse_args()


def load_guidance(path: str) -> list[str]:
    with open(path) as f:
        prompts = json.load(f)
    if not isinstance(prompts, list):
        raise TypeError(f"Expected list in {path}, got {type(prompts)}")
    if len(prompts) < 88:
        raise ValueError(f"Expected at least 88 guidance prompts in {path}, got {len(prompts)}")
    return [str(x) for x in prompts[:88]]


def concept_to_target_class() -> dict[int, int]:
    mapping = {}
    for class_index, concept_indices in DEFAULT_CLASSES_TO_CONCEPTS.items():
        for concept_index in concept_indices:
            mapping[int(concept_index)] = int(class_index)
    return mapping


def binary_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> tuple[float, float, int, int, int]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    intersection = np.logical_and(pred_b, gt_b).sum(dtype=np.int64)
    union = np.logical_or(pred_b, gt_b).sum(dtype=np.int64)
    pred_sum = pred_b.sum(dtype=np.int64)
    gt_sum = gt_b.sum(dtype=np.int64)
    dice = (2.0 * intersection + eps) / (pred_sum + gt_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice), float(iou), int(pred_sum), int(gt_sum), int(intersection)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "eval_guidance_zero_shot.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", buffering=1) as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            run(args, output_dir, log_path)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def run(args: argparse.Namespace, output_dir: Path, log_path: Path) -> None:
    device = torch.device(
        f"cuda:{args.gpu}" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    image_dir = Path(args.data_root) / "images" / args.sequence
    label_dir = Path(args.data_root) / "labels" / args.sequence
    image_paths = sorted(image_dir.glob("*.nii.gz"))
    if args.limit_cases:
        image_paths = image_paths[:args.limit_cases]
    if not image_paths:
        raise RuntimeError(f"No .nii.gz images found in {image_dir}")

    prompts = load_guidance(args.concepts_json)
    target_class_for_concept = concept_to_target_class()
    prompt_chunk_size = max(1, int(args.prompt_chunk_size))

    print(f"Console log: {log_path}")
    print(f"Args: {vars(args)}")
    print(f"Cases: {len(image_paths)}")
    print(f"Guidance prompts: {len(prompts)}")
    print(f"Device: {device}")

    predictor = VoxTellPredictor(model_dir=args.model_dir, device=device)
    reader_writer = NibabelIOWithReorient()

    per_case_path = output_dir / "per_case_prompt_metrics.csv" # 每个病例、每个 guidance 的详细结果
    summary_path = output_dir / "prompt_summary.csv" # 每个 guidance 对自己目标器官的平均 Dice/IoU
    matrix_path = output_dir / "prompt_class_matrix.csv" # 每个 guidance 对 11 个 GT 类的平均 Dice，用来看是否误激活其他器官

    summary = {
        i: {
            "target_dice": [],
            "target_iou": [],
            "target_pred_ratio": [],
            "class_dice": [[] for _ in DEFAULT_CLASSES],
            "class_iou": [[] for _ in DEFAULT_CLASSES],
        }
        for i in range(len(prompts))
    }

    with per_case_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "concept_index",
                "concept",
                "target_class_index",
                "target_class",
                "target_dice",
                "target_iou",
                "pred_voxels",
                "target_gt_voxels",
                "target_intersection",
                "pred_ratio",
            ],
        )
        writer.writeheader()

        for case_id, image_path in enumerate(image_paths, start=1):
            label_path = label_dir / image_path.name
            if not label_path.exists():
                print(f"skip {image_path.name}: missing label {label_path}")
                continue

            print(f"\n[{case_id}/{len(image_paths)}] {image_path.name}")
            image, _ = reader_writer.read_images([str(image_path)])
            label, _ = reader_writer.read_images([str(label_path)])
            label_map = label[0]
            case_voxels = int(np.prod(label_map.shape))

            for start in range(0, len(prompts), prompt_chunk_size):
                end = min(start + prompt_chunk_size, len(prompts))
                chunk_prompts = prompts[start:end]
                preds = predictor.predict_single_image(image, chunk_prompts, output_type="binary")
                for local_index, pred in enumerate(preds):
                    concept_index = start + local_index
                    target_class = target_class_for_concept[concept_index]
                    target_gt = label_map == (target_class + 1)
                    target_dice, target_iou, pred_voxels, gt_voxels, inter = binary_metrics(
                        pred, target_gt
                    )
                    pred_ratio = pred_voxels / max(case_voxels, 1)
                    summary[concept_index]["target_dice"].append(target_dice)
                    summary[concept_index]["target_iou"].append(target_iou)
                    summary[concept_index]["target_pred_ratio"].append(pred_ratio)

                    for class_index in range(len(DEFAULT_CLASSES)):
                        d, j, _, _, _ = binary_metrics(pred, label_map == (class_index + 1))
                        summary[concept_index]["class_dice"][class_index].append(d)
                        summary[concept_index]["class_iou"][class_index].append(j)

                    writer.writerow({
                        "case": image_path.name,
                        "concept_index": concept_index,
                        "concept": prompts[concept_index],
                        "target_class_index": target_class,
                        "target_class": DEFAULT_CLASSES[target_class],
                        "target_dice": target_dice,
                        "target_iou": target_iou,
                        "pred_voxels": pred_voxels,
                        "target_gt_voxels": gt_voxels,
                        "target_intersection": inter,
                        "pred_ratio": pred_ratio,
                    })

                del preds

    with summary_path.open("w", newline="") as f:
        fieldnames = [
            "concept_index",
            "concept",
            "target_class_index",
            "target_class",
            "mean_target_dice",
            "mean_target_iou",
            "mean_pred_ratio",
            "best_matched_class",
            "best_matched_dice",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for concept_index, prompt in enumerate(prompts):
            target_class = target_class_for_concept[concept_index]
            class_dice_means = [
                float(np.mean(summary[concept_index]["class_dice"][class_index]))
                for class_index in range(len(DEFAULT_CLASSES))
            ]
            best_class = int(np.argmax(class_dice_means))
            row = {
                "concept_index": concept_index,
                "concept": prompt,
                "target_class_index": target_class,
                "target_class": DEFAULT_CLASSES[target_class],
                "mean_target_dice": float(np.mean(summary[concept_index]["target_dice"])),
                "mean_target_iou": float(np.mean(summary[concept_index]["target_iou"])),
                "mean_pred_ratio": float(np.mean(summary[concept_index]["target_pred_ratio"])),
                "best_matched_class": DEFAULT_CLASSES[best_class],
                "best_matched_dice": class_dice_means[best_class],
            }
            writer.writerow(row)

    with matrix_path.open("w", newline="") as f:
        fieldnames = ["concept_index", "concept", "target_class"] + [
            f"dice_{name}" for name in DEFAULT_CLASSES
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for concept_index, prompt in enumerate(prompts):
            target_class = target_class_for_concept[concept_index]
            row = {
                "concept_index": concept_index,
                "concept": prompt,
                "target_class": DEFAULT_CLASSES[target_class],
            }
            for class_index, class_name in enumerate(DEFAULT_CLASSES):
                row[f"dice_{class_name}"] = float(
                    np.mean(summary[concept_index]["class_dice"][class_index])
                )
            writer.writerow(row)

    print("\nTop guidance prompts by target Dice:")
    ranked = []
    for concept_index, prompt in enumerate(prompts):
        target_class = target_class_for_concept[concept_index]
        ranked.append((
            float(np.mean(summary[concept_index]["target_dice"])),
            concept_index,
            DEFAULT_CLASSES[target_class],
            prompt,
        ))
    for dice, concept_index, target_class, prompt in sorted(ranked, reverse=True)[:20]:
        print(f"  [{concept_index:02d}] {target_class:20s} dice={dice:.4f} prompt={prompt}")

    print(f"\nWrote: {per_case_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {matrix_path}")


if __name__ == "__main__":
    main()

'''
python -m voxtell.eval_guidance_zero_shot
'''