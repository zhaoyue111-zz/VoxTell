import numpy as np
import nibabel as nib
from pathlib import Path

def global_unique_values(folder: str, recursive: bool = True):
    folder = Path(folder)
    pattern = "**/*.nii.gz" if recursive else "*.nii.gz"

    all_vals = set()
    for fp in folder.glob(pattern):
        data = nib.load(str(fp)).get_fdata(dtype=np.float32)
        vals = np.unique(np.rint(data).astype(np.int64))
        all_vals.update(vals.tolist())
    return sorted(all_vals)

if __name__ == "__main__":
    folder = r"D:\dataset\CT_MRI_DATA_3D\labels\T2"
    print(global_unique_values(folder))
    # EAP: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # Delay: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # P0: [0, 5]
    # P1: [0, 5]
    # PreArtery: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # PV: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # T2: [0, 5]