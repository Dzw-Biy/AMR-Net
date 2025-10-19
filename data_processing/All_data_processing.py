import os
import re
import numpy as np
from GPS_processing import getCoord
from image_processing import getImg
from Lidar_processing import getLidar
from Gray_processing import getGray



np.random.seed(42)

# Load all modalities
coord_data, coord_label, coord_quality, coord_name = getCoord()
img_data, img_label, img_quality, img_name = getImg()
lidar_data, lidar_label, lidar_quality, lidar_name = getLidar()
gray_data, gray_label, gray_quality, gray_name = getGray()

num_samples = coord_data.shape[0]


def extract_idx(fname):
    """Extract numeric index from filename for sorting."""
    match = re.search(r'(\d+)', fname)
    return int(match.group(1)) if match else -1


data_dir = '<DATA_DIR>/mmWave_data'  # <-- Replace with actual folder

file_list = sorted(
    [f for f in os.listdir(data_dir) if f.endswith('.txt')],
    key=extract_idx
)

power_list = []
for filename in file_list:
    file_path = os.path.join(data_dir, filename)
    values = np.loadtxt(file_path)
    if values.size != 64:
        raise ValueError(f"{filename}: expected 64 values, got {values.size}")
    power_list.append(values)

power_data = np.stack(power_list, axis=0)


indices = np.random.permutation(num_samples)
train_end = int(0.8 * num_samples)
val_end = train_end + int(0.1 * num_samples)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]


def save_modality(modality_name, base_dir, data, label=None, quality=None, name_list=None,
                  train_idx=None, val_idx=None, test_idx=None):
    """
    Save train/val/test split for a given modality.
    - For modalities with label/quality/name → save all
    - For 'power' (no labels) → save only data
    """
    os.makedirs(f"{base_dir}/{modality_name}", exist_ok=True)

    def save_split(split, idx):
        subset = data[idx]
        save_path = f"{base_dir}/{modality_name}/{split}.npz"

        if label is not None:
            np.savez(save_path,
                     data=subset,
                     label=label[idx],
                     quality=quality[idx],
                     name=[name_list[i] for i in idx])
        else:
            np.savez(save_path, data=subset)

    save_split("train", train_idx)
    save_split("val", val_idx)
    save_split("test", test_idx)


output_root = "<OUTPUT_DIR>/scenario8"

# Power is used for subsequent RPR metric calculations.
save_modality("power", f"{output_root}/p50",
              data=power_data,
              train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

# Coord / LiDAR / Image / Gray
save_modality("coord", f"{output_root}/normal",
              data=coord_data, label=coord_label, quality=coord_quality, name_list=coord_name,
              train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

save_modality("lidar", f"{output_root}/snowy",
              data=lidar_data, label=lidar_label, quality=lidar_quality, name_list=lidar_name,
              train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

save_modality("img", f"{output_root}/night",
              data=img_data, label=img_label, quality=img_quality, name_list=img_name,
              train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

save_modality("gray", f"{output_root}/night",
              data=gray_data, label=gray_label, quality=gray_quality, name_list=gray_name,
              train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

print(" All modalities (coord, img, lidar, gray, power) saved successfully.")