import os
import nibabel as nib
import numpy as np

# 修改为你的文件夹路径
# data_dir = "/path/to/nii_folder"
img="/home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/WCH_MREI145_predict/IN/0000661245.nii.gz"

img=nib.load(img)
data=np.asanyarray(img.dataobj)
unique_values = np.unique(data)
print(unique_values)

# all_labels = set()
# for filename in sorted(os.listdir(data_dir)):
#     if not filename.endswith(".nii.gz"):
#         continue
#
#     file_path = os.path.join(data_dir, filename)
#
#     # 读取 NIfTI 文件
#     img = nib.load(file_path)
#     data = np.asanyarray(img.dataobj)
#
#     # 获取唯一值
#     unique_values = np.unique(data)
#     all_labels.update(unique_values.tolist())
#
#     print(f"{filename}: {unique_values}")
#
# print("\n所有文件的唯一标签值：")
# print(sorted(all_labels))