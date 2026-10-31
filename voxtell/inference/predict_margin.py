import nibabel as nib
import csv
from pathlib import Path
import numpy as np
import torch

from voxtell.inference.predict_from_raw_data import get_reader_writer, save_segmentation
from voxtell.inference.predictor import VoxTellPredictor

def get_bbox_from_mask(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    start = coords.min(axis=0).astype(int)
    end = (coords.max(axis=0) + 1).astype(int)
    return start, end


def bbox_contains(outer_bbox, inner_bbox):
    outer_start, outer_end = outer_bbox
    inner_start, inner_end = inner_bbox

    return bool(np.all(outer_start <= inner_start) and np.all(outer_end >= inner_end))


def expand_bbox(start, end, margin, shape):
    start = np.asarray(start, dtype=int)
    end = np.asarray(end, dtype=int)
    shape = np.asarray(shape, dtype=int)

    start = np.maximum(start - margin, 0)
    end = np.minimum(end + margin, shape)

    return start, end


def build_final_liver_bbox(liver_bbox, tumor_bbox, volume_shape, extra_margin=8):
    if liver_bbox is None:
        if tumor_bbox is None:
            return None, "empty_liver_and_tumor"

        tumor_start, tumor_end = tumor_bbox
        final_start, final_end = expand_bbox(tumor_start, tumor_end, extra_margin, volume_shape)
        return (final_start, final_end), "liver_empty_use_tumor"

    if tumor_bbox is None:
        return liver_bbox, "tumor_empty_keep_liver"

    if bbox_contains(liver_bbox, tumor_bbox):
        return liver_bbox, "liver_already_contains_tumor"

    liver_start, liver_end = liver_bbox
    tumor_start, tumor_end = tumor_bbox

    cover_start = np.minimum(liver_start, tumor_start)
    cover_end = np.maximum(liver_end, tumor_end)

    final_start, final_end = expand_bbox(cover_start, cover_end, extra_margin, volume_shape)

    return (final_start, final_end), "expanded_to_cover_tumor_plus_8"


def crop_array_dhw(array, bbox):
    start, end = bbox
    d0, h0, w0 = start
    d1, h1, w1 = end

    return array[d0:d1, h0:h1, w0:w1]


def save_cropped_nifti(source_path, output_path, bbox_dhw, force_binary=False):
    source = nib.as_closest_canonical(nib.load(str(source_path)))
    data = np.asarray(source.dataobj)

    if data.ndim != 3:
        raise ValueError(f"{source_path}: 目前只支持 3D NIfTI，实际 shape={data.shape}")

    start_dhw, end_dhw = bbox_dhw

    d0, h0, w0 = [int(v) for v in start_dhw]
    d1, h1, w1 = [int(v) for v in end_dhw]

    x0, y0, z0 = w0, h0, d0
    x1, y1, z1 = w1, h1, d1

    cropped = data[x0:x1, y0:y1, z0:z1]

    if force_binary:
        cropped = (cropped > 0).astype(np.uint8)

    new_affine = source.affine.copy()
    offset_xyz = np.asarray([x0, y0, z0, 1.0])
    new_origin = source.affine @ offset_xyz
    new_affine[:3, 3] = new_origin[:3]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = nib.Nifti1Image(cropped, new_affine, source.header.copy())

    if force_binary:
        output.set_data_dtype(np.uint8)

    nib.save(output, str(output_path))


def save_numpy_mask_as_cropped_nifti(mask_dhw, source_image_path, output_path, bbox_dhw):
    source = nib.as_closest_canonical(nib.load(str(source_image_path)))

    cropped_dhw = crop_array_dhw(mask_dhw.astype(np.uint8), bbox_dhw)
    cropped_xyz = cropped_dhw.transpose(2, 1, 0)

    start_dhw, _ = bbox_dhw
    d0, h0, w0 = [int(v) for v in start_dhw]

    x0, y0, z0 = w0, h0, d0

    new_affine = source.affine.copy()
    offset_xyz = np.asarray([x0, y0, z0, 1.0])
    new_origin = source.affine @ offset_xyz
    new_affine[:3, 3] = new_origin[:3]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = nib.Nifti1Image(cropped_xyz, new_affine)
    output.set_data_dtype(np.uint8)

    nib.save(output, str(output_path))


def generate_liver_cuboid_roi():
    prompts = ["liver"]
    sequences = ["HBP"]
    device = torch.device("cuda:0")

    model_path = Path("/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/model")
    text_model = "/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/model/Qwen3-Embedding-4B"

    predictor = VoxTellPredictor(model_dir=str(model_path), text_encoding_model=text_model, device=device)

    data_root = Path("/data/hilab/datasets/WCH_MRI145")
    output_root = Path("/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/WCH_MREI145_predict/liver")

    extra_margin = 8
    tumor_labels = (1,)

    for sequence in sequences:
        input_dir = data_root / sequence / "images"
        tumor_dir = data_root / sequence / "labels"

        sequence_output = output_root / sequence
        image_output = sequence_output / "images"
        tumor_output = sequence_output / "labels"
        liver_output = sequence_output / "liver_masks"
        full_liver_output = sequence_output / "full_liver_prediction"

        image_output.mkdir(parents=True, exist_ok=True)
        tumor_output.mkdir(parents=True, exist_ok=True)
        liver_output.mkdir(parents=True, exist_ok=True)
        full_liver_output.mkdir(parents=True, exist_ok=True)

        filenames = sorted(p.name for p in input_dir.glob("*.nii.gz") if p.is_file())

        if not filenames:
            print(f"{sequence}: 没有找到 .nii.gz 文件")
            continue

        csv_path = sequence_output / "roi_info.csv"

        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            writer.writerow([
                "filename",
                "status",
                "d0",
                "d1",
                "h0",
                "h1",
                "w0",
                "w1",
                "size_d",
                "size_h",
                "size_w",
                "liver_voxels",
                "tumor_voxels",
                "tumor_inside_original_liver_ratio",
            ])

            for index, filename in enumerate(filenames, start=1):
                image_path = input_dir / filename
                tumor_path = tumor_dir / filename

                if not tumor_path.is_file():
                    raise FileNotFoundError(f"缺少 tumor mask: {tumor_path}")

                reader_writer = get_reader_writer(str(image_path))
                img, props = reader_writer.read_images([str(image_path)])

                with torch.inference_mode():
                    segmentation = predictor.predict_single_image(img, prompts)

                save_segmentation(segmentation[0], full_liver_output, filename, props, suffix="nii.gz")

                liver_pred = np.squeeze(np.asarray(segmentation)) > 0

                tumor_gt, _ = reader_writer.read_images([str(tumor_path)])
                tumor_gt = np.squeeze(np.asarray(tumor_gt))
                tumor_gt = np.rint(tumor_gt).astype(np.int64)
                tumor_mask = np.isin(tumor_gt, tumor_labels)

                if liver_pred.shape != tumor_mask.shape:
                    raise ValueError(f"{filename}: liver shape {liver_pred.shape} != tumor shape {tumor_mask.shape}")

                liver_bbox = get_bbox_from_mask(liver_pred)
                tumor_bbox = get_bbox_from_mask(tumor_mask)

                final_bbox, status = build_final_liver_bbox(liver_bbox, tumor_bbox, liver_pred.shape, extra_margin=extra_margin)

                if final_bbox is None:
                    print(f"[{sequence}] {filename}: liver 和 tumor 均为空，跳过")
                    continue

                liver_voxels = int(liver_pred.sum())
                tumor_voxels = int(tumor_mask.sum())

                if tumor_voxels > 0:
                    tumor_inside_liver = int(np.logical_and(tumor_mask, liver_pred).sum())
                    tumor_inside_ratio = tumor_inside_liver / tumor_voxels
                else:
                    tumor_inside_ratio = np.nan

                final_start, final_end = final_bbox
                roi_size = final_end - final_start

                save_cropped_nifti(image_path, image_output / filename, final_bbox, force_binary=False)
                save_cropped_nifti(tumor_path, tumor_output / filename, final_bbox, force_binary=False)
                save_numpy_mask_as_cropped_nifti(liver_pred, image_path, liver_output / filename, final_bbox)

                writer.writerow([
                    filename,
                    status,
                    final_start[0],
                    final_end[0],
                    final_start[1],
                    final_end[1],
                    final_start[2],
                    final_end[2],
                    roi_size[0],
                    roi_size[1],
                    roi_size[2],
                    liver_voxels,
                    tumor_voxels,
                    tumor_inside_ratio,
                ])

                f.flush()

                print(
                    f"[{sequence}] {index}/{len(filenames)} {filename} | "
                    f"status={status} | "
                    f"original tumor coverage={tumor_inside_ratio:.4f} | "
                    f"ROI size={tuple(roi_size.tolist())}"
                )

        print(f"\n[{sequence}] liver cuboid ROI 已保存到: {sequence_output}")
        print(f"[{sequence}] ROI 信息: {csv_path}\n")

if __name__ == '__main__':
    generate_liver_cuboid_roi()