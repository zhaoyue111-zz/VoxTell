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
    prompts = ["spleen", "right_kidney", "left_kidney", "gallbladder", "liver", "stomach", "esophagus",
               "inferior_vena_cava", "pancreas", "duodenum","aotra"]
    print("\nLabel mapping:")
    for i, prompt in enumerate(prompts):
        print(f"  {i + 1}: {prompt}")

    device = torch.device(f'cuda:0')
    model_path = Path("/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/model")
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

def evaluate_saved_predictions():
    prompts = ["liver"]
    sequences = ["DWI"]

    data_root = Path("/data/hilab/datasets/WCH_MRI145")
    output_folder = Path("/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/WCH_MREI145_predict/liver")

    for sequence in sequences:
        mask_dir = data_root / sequence / "liver"
        pred_dir = output_folder / sequence

        filenames = sorted(p.name for p in pred_dir.glob("*.nii.gz") if p.is_file())

        if not filenames:
            print(f"{sequence}: 没有找到预测结果")
            continue

        metric_values = []
        csv_path = pred_dir / "metrics.csv"

        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "prompt", "dice", "iou", "hd", "asd"])

            for filename in filenames:
                pred_path = pred_dir / filename
                gt_path = mask_dir / filename

                if not gt_path.is_file():
                    print(f"跳过，缺少 GT：{gt_path}")
                    continue

                pred_reader_writer = get_reader_writer(str(pred_path))
                pred, pred_props = pred_reader_writer.read_images([str(pred_path)])

                gt_reader_writer = get_reader_writer(str(gt_path))
                gt, gt_props = gt_reader_writer.read_images([str(gt_path)])

                pred_3d = np.asarray(pred)
                gt_3d = np.asarray(gt)

                if pred_3d.shape != gt_3d.shape:
                    raise ValueError(f"{filename}: 预测形状 {pred_3d.shape} 与 GT 形状 {gt_3d.shape} 不一致")

                dice, iou = dice_iou(pred_3d, gt_3d)
                hd, asd_val = compute_boundary_metrics(pred_3d,gt_3d,spacing=gt_props["spacing"])

                values = [float(dice),float(iou),float(hd),float(asd_val)]
                metric_values.append(values)

                writer.writerow([filename, prompts[0], *values])
                f.flush()

                print(f"[{sequence}] {filename} dice: {dice:.4f}, iou: {iou:.4f}, hd: {hd:.4f}, asd: {asd_val:.4f}")

            if not metric_values:
                print(f"{sequence}: 没有成功计算任何病例")
                continue

            means = np.mean(np.asarray(metric_values), axis=0)
            writer.writerow(["MEAN", prompts[0], *means])

        print(
            f"\n[{sequence}] n={len(metric_values)}, "
            f"dice: {means[0]:.4f}, "
            f"iou: {means[1]:.4f}, "
            f"hd: {means[2]:.4f}, "
            f"asd: {means[3]:.4f}, "
            f"CSV: {csv_path}\n"
        )

def predict_single():
    prompts = ["liver"]
    sequences = ["AP", "DWI", "IN", "OPP", "PRE", "PVP", "T2WI", "TP"]
    # sequences=['HBP']
    device = torch.device("cuda:0")
    model_path = Path("/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/model")
    predictor = VoxTellPredictor(model_dir=str(model_path),text_encoding_model="/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/model/Qwen3-Embedding-4B",device=device)

    data_root = Path("/data/hilab/datasets/WCH_MRI145")
    output_folder = Path("/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/WCH_MREI145_predict/liver")
    output_folder.mkdir(parents=True, exist_ok=True)

    for sequence in sequences:
        input_dir = data_root / sequence / "images"
        mask_dir = data_root / sequence / "liver"
        sub_output = output_folder / sequence
        sub_output.mkdir(parents=True, exist_ok=True)

        filenames = sorted(p.name for p in input_dir.glob("*.nii.gz") if p.is_file())
        if not filenames:
            print(f"{sequence}: 没有找到 .nii.gz 文件")
            continue

        # 每个序列单独统计
        metric_values = []
        csv_path = sub_output / "metrics.csv"

        # w 表示每次运行覆盖该序列原有的 CSV
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "prompt", "dice", "iou", "hd", "asd"])
            f.flush()

            for filename in filenames:
                image_path = input_dir / filename
                gt_path = mask_dir / filename

                if not gt_path.is_file():
                    raise FileNotFoundError(f"缺少 GT: {gt_path}")

                reader_writer = get_reader_writer(str(image_path))
                img, props = reader_writer.read_images([str(image_path)])

                with torch.inference_mode():
                    segmentation = predictor.predict_single_image(img, prompts)  # [1, D, H, W]
                save_segmentation(segmentation[0],sub_output,filename,props,suffix="nii.gz")

                gt, _ = reader_writer.read_images([str(gt_path)])

                # 单个 prompt：取出对应的三维预测和 GT
                pred_3d = np.asarray(segmentation)
                gt_3d = np.asarray(gt)

                if pred_3d.shape != gt_3d.shape:
                    raise ValueError(f"{filename}: 预测形状 {pred_3d.shape}与 GT 形状 {gt_3d.shape} 不一致")

                dice, iou = dice_iou(pred_3d, gt_3d)
                hd, asd_val = compute_boundary_metrics(pred_3d,gt_3d,spacing=props["spacing"])

                values = [float(dice),float(iou),float(hd),float(asd_val)]
                metric_values.append(values)

                writer.writerow([filename, prompts[0], *values])
                f.flush()  # 已完成病例立即写入文件

                print(f"[{sequence}] {filename} dice: {dice:.4f}, iou: {iou:.4f}, hd: {hd:.4f}, asd: {asd_val:.4f}")

            means = np.mean(np.asarray(metric_values), axis=0)
            writer.writerow(["MEAN", prompts[0], *means])

        print(f"\n[{sequence}] n={len(metric_values)}, dice: {means[0]:.4f}, iou: {means[1]:.4f}, hd: {means[2]:.4f}, asd: {means[3]:.4f}, CSV: {csv_path}\n")

def predict_single_192():
    prompts = ["liver cancer"]
    device = torch.device("cuda:0")

    model_path = Path(
        "/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/model"
    )
    predictor = VoxTellPredictor(
        model_dir=str(model_path),
        text_encoding_model=str(
            model_path / "Qwen3-Embedding-4B"
        ),
        device=device,
    )
    predictor.network.eval()

    # preprocess_voxtell_192.py生成的npz目录
    preprocessed_dir = Path(
        "/home/ubuntu/zy/pythoncode/sfda/CM-TTA/"
        "output/voxtell_192/HBP/preprocess/cases"
    )

    output_folder = Path(
        "/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/"
        "WCH_MRI145_predict_192"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(preprocessed_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"没有找到预处理后的npz文件：{preprocessed_dir}"
        )

    # 固定预训练文本特征，不进行TTA
    with torch.inference_mode():
        text_feature = predictor.embed_text_prompts(prompts).detach()

    # 当前实验要求VoxTell窗口为192^3
    patch_size = tuple(int(v) for v in predictor.patch_size)
    if patch_size != (192, 192, 192):
        raise ValueError(
            f"VoxTell patch_size={patch_size}，预期为(192,192,192)"
        )

    metric_values = []
    csv_path = output_folder / "metrics_192.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "prompt",
            "dice",
            "iou",
            "pred_foreground_voxels",
            "gt_foreground_voxels",
        ])
        f.flush()

        for index, npz_path in enumerate(npz_files, start=1):
            with np.load(npz_path, allow_pickle=False) as archive:
                image_192 = archive["image_192"].astype(
                    np.float32, copy=False
                )
                gt_192 = archive["gt_192"].astype(
                    np.uint8, copy=False
                )
                case_id = str(archive["case_id"].item())

            if image_192.shape != (1, 192, 192, 192):
                raise ValueError(
                    f"{npz_path.name}: image_192形状错误："
                    f"{image_192.shape}"
                )
            if gt_192.shape != (1, 192, 192, 192):
                raise ValueError(
                    f"{npz_path.name}: gt_192形状错误："
                    f"{gt_192.shape}"
                )

            image_tensor = torch.from_numpy(
                image_192
            ).float().contiguous()

            # image_192已经完成裁剪、标准化和缩放，
            # 不再调用predict_single_image/preprocess
            with torch.inference_mode():
                logits = predictor.predict_sliding_window_return_logits(
                    image_tensor,
                    text_feature.to(predictor.device),
                ).float().cpu()

                probability = torch.sigmoid(logits)

            # 去除前导的batch/prompt单例维度，得到[D,H,W]
            while (
                probability.ndim > 3
                and probability.shape[0] == 1
            ):
                probability = probability[0]

            if tuple(probability.shape) != (192, 192, 192):
                raise ValueError(
                    f"{case_id}: 输出形状错误："
                    f"{tuple(probability.shape)}"
                )

            pred_3d = (
                probability.numpy() >= 0.5
            ).astype(np.uint8)
            gt_3d = gt_192[0]

            dice, iou = dice_iou(pred_3d[None, ...], gt_192)

            pred_voxels = int(np.count_nonzero(pred_3d))
            gt_voxels = int(np.count_nonzero(gt_3d))

            metric_values.append([
                float(dice),
                float(iou),
            ])

            writer.writerow([
                f"{case_id}.nii.gz",
                prompts[0],
                float(dice),
                float(iou),
                pred_voxels,
                gt_voxels,
            ])
            f.flush()

            # 192^3结果直接保存为npy，避免错误套用原图affine
            np.save(
                output_folder / f"{case_id}_pred_192.npy",
                pred_3d,
            )
            np.save(
                output_folder / f"{case_id}_probability_192.npy",
                probability.numpy().astype(np.float32),
            )

            print(
                f"[{index}/{len(npz_files)}] {case_id}: "
                f"dice={dice:.4f}, iou={iou:.4f}, "
                f"pred_fg={pred_voxels}, gt_fg={gt_voxels}"
            )

        means = np.mean(
            np.asarray(metric_values, dtype=np.float64),
            axis=0,
        )
        writer.writerow([
            "MEAN",
            prompts[0],
            float(means[0]),
            float(means[1]),
            "",
            "",
        ])

    print(
        f"\nn={len(metric_values)}, "
        f"dice={means[0]:.4f}, "
        f"iou={means[1]:.4f}, "
        f"CSV={csv_path}"
    )

import csv
from pathlib import Path

import numpy as np
import torch


def _bbox_from_mask(mask: np.ndarray):
    """
    mask: [D, H, W] bool
    return:
        start: [d0, h0, w0]
        end:   [d1, h1, w1]，右开区间
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    start = coords.min(axis=0)
    end = coords.max(axis=0) + 1
    return start.astype(int), end.astype(int)


def _expand_bbox(start, end, margin, shape):
    start = np.asarray(start, dtype=int)
    end = np.asarray(end, dtype=int)
    shape = np.asarray(shape, dtype=int)

    start = np.maximum(start - margin, 0)
    end = np.minimum(end + margin, shape)

    return start, end


def _required_margin_to_cover(liver_bbox, tumor_bbox):
    """
    计算从 liver bbox 向六个方向做“统一对称扩张”时，
    至少需要多少 voxel 才能完整包含 tumor bbox。
    """
    liver_start, liver_end = liver_bbox
    tumor_start, tumor_end = tumor_bbox

    need_before = np.maximum(liver_start - tumor_start, 0)
    need_after = np.maximum(tumor_end - liver_end, 0)

    # 六个方向中最大的扩张量
    required_margin = int(max(need_before.max(), need_after.max()))

    return required_margin, need_before, need_after


def predict_liver_and_check_tumor_roi():
    prompts = ["liver"]
    sequences = ["HBP"]

    device = torch.device("cuda:0")

    model_path = Path(
        "/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/model"
    )
    text_model = (
        "/home/ubuntu/zy/pythoncode/sfda/"
        "VoxTell_from_disk/model/Qwen3-Embedding-4B"
    )

    predictor = VoxTellPredictor(
        model_dir=str(model_path),
        text_encoding_model=text_model,
        device=device,
    )

    data_root = Path("/data/hilab/datasets/WCH_MRI145")

    output_root = Path(
        "/home/ubuntu/zy/pythoncode/sfda/"
        "VoxTell_from_disk/WCH_MREI145_predict/liver_roi"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    # “完全包住 tumor 后，再额外扩 10 voxel”
    extra_margin = 10

    # 根据你的 tumor GT 标签修改。
    # 如果当前 label 是二值 tumor mask，保持 (1,) 即可。
    tumor_labels = (1,)

    for sequence in sequences:
        input_dir = data_root / sequence / "images"
        tumor_mask_dir = data_root / sequence / "labels"

        sequence_output = output_root / sequence
        liver_output = sequence_output / "liver_prediction"

        sequence_output.mkdir(parents=True, exist_ok=True)
        liver_output.mkdir(parents=True, exist_ok=True)

        filenames = sorted(
            p.name for p in input_dir.glob("*.nii.gz") if p.is_file()
        )

        if not filenames:
            print(f"{sequence}: 没有找到 .nii.gz 文件")
            continue

        csv_path = sequence_output / "liver_tumor_roi_check.csv"

        with csv_path.open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.writer(f)

            writer.writerow([
                "filename",
                "liver_empty",
                "tumor_empty",

                "tumor_voxels",
                "tumor_inside_liver_voxels",
                "tumor_inside_liver_ratio",

                "required_margin",

                "liver_d0", "liver_d1",
                "liver_h0", "liver_h1",
                "liver_w0", "liver_w1",

                "oracle_d0", "oracle_d1",
                "oracle_h0", "oracle_h1",
                "oracle_w0", "oracle_w1",

                "oracle_size_d",
                "oracle_size_h",
                "oracle_size_w",
            ])

            for index, filename in enumerate(filenames, start=1):
                image_path = input_dir / filename
                tumor_path = tumor_mask_dir / filename

                if not tumor_path.is_file():
                    raise FileNotFoundError(
                        f"缺少 tumor GT: {tumor_path}"
                    )

                reader_writer = get_reader_writer(str(image_path))

                img, props = reader_writer.read_images(
                    [str(image_path)]
                )

                # img: [C, D, H, W]
                with torch.inference_mode():
                    segmentation = predictor.predict_single_image(
                        img,
                        prompts,
                    )

                # 保存原始 liver prediction
                save_segmentation(
                    segmentation[0],
                    liver_output,
                    filename,
                    props,
                    suffix="nii.gz",
                )

                # segmentation 通常 [1, D, H, W]
                liver_pred = np.asarray(segmentation)
                liver_pred = np.squeeze(liver_pred) > 0

                tumor_gt, _ = reader_writer.read_images(
                    [str(tumor_path)]
                )
                tumor_gt = np.squeeze(np.asarray(tumor_gt))

                tumor_mask = np.isin(
                    np.rint(tumor_gt).astype(np.int64),
                    tumor_labels,
                )

                if liver_pred.shape != tumor_mask.shape:
                    raise ValueError(
                        f"{filename}: liver shape "
                        f"{liver_pred.shape} != tumor shape "
                        f"{tumor_mask.shape}"
                    )

                liver_bbox = _bbox_from_mask(liver_pred)
                tumor_bbox = _bbox_from_mask(tumor_mask)

                liver_empty = liver_bbox is None
                tumor_empty = tumor_bbox is None

                tumor_voxels = int(tumor_mask.sum())

                if tumor_voxels > 0:
                    tumor_inside = int(
                        np.logical_and(
                            tumor_mask,
                            liver_pred,
                        ).sum()
                    )
                    coverage = tumor_inside / tumor_voxels
                else:
                    tumor_inside = 0
                    coverage = np.nan

                if liver_bbox is None:
                    # 该病例 liver 模型完全失败。
                    # 这里只做 oracle 诊断，不能用于正式 SFDA。
                    if tumor_bbox is None:
                        print(
                            f"[{sequence}] {filename}: "
                            "liver 和 tumor 都为空"
                        )
                        continue

                    required_margin = -1

                    oracle_start, oracle_end = _expand_bbox(
                        tumor_bbox[0],
                        tumor_bbox[1],
                        extra_margin,
                        liver_pred.shape,
                    )

                    liver_start = np.array([-1, -1, -1])
                    liver_end = np.array([-1, -1, -1])

                else:
                    liver_start, liver_end = liver_bbox

                    if tumor_bbox is None:
                        required_margin = 0

                    else:
                        required_margin, _, _ = (
                            _required_margin_to_cover(
                                liver_bbox,
                                tumor_bbox,
                            )
                        )

                    # 完全覆盖 tumor 所需 margin + 再额外 10 voxel
                    final_margin = (
                        required_margin + extra_margin
                    )

                    oracle_start, oracle_end = _expand_bbox(
                        liver_start,
                        liver_end,
                        final_margin,
                        liver_pred.shape,
                    )

                oracle_size = oracle_end - oracle_start

                writer.writerow([
                    filename,
                    int(liver_empty),
                    int(tumor_empty),

                    tumor_voxels,
                    tumor_inside,
                    coverage,

                    required_margin,

                    *[
                        liver_start[0],
                        liver_end[0],
                        liver_start[1],
                        liver_end[1],
                        liver_start[2],
                        liver_end[2],
                    ],

                    *[
                        oracle_start[0],
                        oracle_end[0],
                        oracle_start[1],
                        oracle_end[1],
                        oracle_start[2],
                        oracle_end[2],
                    ],

                    *oracle_size.tolist(),
                ])

                f.flush()

                print(
                    f"[{sequence}] "
                    f"{index}/{len(filenames)} "
                    f"{filename} | "
                    f"tumor coverage={coverage:.4f} | "
                    f"required_margin={required_margin} | "
                    f"oracle crop={tuple(oracle_size)}"
                )

        print(
            f"\n[{sequence}] ROI 检查完成: {csv_path}\n"
        )

if __name__ == '__main__':
    os.environ['HF_HUB_OFFLINE'] = '1'
    evaluate_saved_predictions()
    # predict_single()
    # predict_single_192()
