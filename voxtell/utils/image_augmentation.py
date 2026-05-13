from __future__ import annotations

from typing import Tuple

import numpy as np
import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform


def apply_contrast_enhancement(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Apply simple contrast enhancement around the mean intensity.

    Args:
        image: Input image array with shape (Z, Y, X) or (C, Z, Y, X).
        factor: Contrast scaling factor (>0). 1.0 keeps the image unchanged.

    Returns:
        Contrast-enhanced image array (float32).
    """
    if factor <= 0:
        raise ValueError(f"contrast factor must be > 0, got {factor}")

    image = image.astype(np.float32, copy=True)

    if image.ndim == 3:
        return _enhance_channel(image, factor)

    if image.ndim == 4:
        enhanced = np.empty_like(image, dtype=np.float32)
        for channel in range(image.shape[0]):
            enhanced[channel] = _enhance_channel(image[channel], factor)
        return enhanced

    raise ValueError(f"image must be 3D or 4D (C, Z, Y, X), got shape {image.shape}")


def _enhance_channel(channel: np.ndarray, factor: float) -> np.ndarray:
    mean_val = float(channel.mean())
    min_val = float(channel.min())
    max_val = float(channel.max())
    enhanced = mean_val + factor * (channel - mean_val)
    return np.clip(enhanced, min_val, max_val, out=enhanced)


def save_reoriented_nifti(image: np.ndarray, output_fname: str, properties: dict) -> None:
    """
    Save a NIfTI image in the original orientation using properties from NibabelIOWithReorient.

    Args:
        image: Image array in nnUNet/NibabelIOWithReorient layout.
        output_fname: Output file path.
        properties: Properties returned by NibabelIOWithReorient.read_images.
    """
    if "nibabel_stuff" not in properties:
        raise ValueError("properties missing nibabel_stuff metadata required for reorientation")

    nib_stuff = properties["nibabel_stuff"]
    original_affine = nib_stuff.get("original_affine")
    reoriented_affine = nib_stuff.get("reoriented_affine")
    if original_affine is None or reoriented_affine is None:
        raise ValueError("properties missing original_affine or reoriented_affine for saving")

    image = image.astype(np.float32, copy=True)
    image_to_save = _to_nibabel_layout(image)

    img_nib = nib.Nifti1Image(image_to_save, affine=reoriented_affine)
    img_ornt = io_orientation(original_affine)
    ras_ornt = axcodes2ornt("RAS")
    from_canonical = ornt_transform(ras_ornt, img_ornt)
    img_nib_reoriented = img_nib.as_reoriented(from_canonical)
    nib.save(img_nib_reoriented, output_fname)


def _to_nibabel_layout(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return image.transpose((2, 1, 0))

    if image.ndim == 4:
        if image.shape[0] == 1:
            return image[0].transpose((2, 1, 0))
        return image.transpose((3, 2, 1, 0))

    raise ValueError(f"image must be 3D or 4D (C, Z, Y, X), got shape {image.shape}")
