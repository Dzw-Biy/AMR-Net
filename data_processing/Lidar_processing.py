import os
import numpy as np
import pandas as pd
import scipy.io as sio
# from Quality_acess.lidar_quality_2D import compute_confusion_score   # optional quality module



def getLidar():
    """
    Load LiDAR frames from .mat files, normalize (distance, angle),
    and compute quality scores.

    Returns:
        processed_data : np.ndarray [N, T, 2]
        labels_array   : np.ndarray [N,]
        quality_array  : np.ndarray [N,]
        file_list      : list[str]
    """

    csv_file = "<DATA_DIR>/scenario8.csv"
    base_dir = "<DATA_DIR>/scenario8/lidar_weather/snowy/"

    df = pd.read_csv(csv_file)
    lidar_paths = df["unit1_lidar_SCR"].tolist()
    labels = df["unit1_beam_index"].tolist()

    processed_data = []
    valid_labels = []
    quality_scores = []
    file_list = []

    for path, label in zip(lidar_paths, labels):
        file_name = os.path.basename(path)
        full_path = os.path.join(base_dir, path.lstrip("./"))

        if not os.path.exists(full_path):
            print(f"[WARN] File not found: {full_path}")
            continue

        try:
            mat_data = sio.loadmat(full_path)
            lidar_points = mat_data["data"]

            # optional quality evaluation
            try:
                from Quality_acess.lidar_quality_2D import compute_confusion_score
                quality_score = compute_confusion_score(lidar_points)
            except Exception:
                quality_score = 0.8  # default fallback

            # verify shape
            if lidar_points.ndim != 2 or lidar_points.shape[1] != 2:
                raise ValueError(f"{file_name}: invalid LiDAR shape {lidar_points.shape}")

            distances, angles = lidar_points[:, 0], lidar_points[:, 1]

            # --- normalize distance ---
            d_min, d_max = np.min(distances), np.max(distances)
            if d_max != d_min:
                dist_norm = (distances - d_min) / (d_max - d_min)
            else:
                dist_norm = np.zeros_like(distances)
                print(f"[WARN] Constant distance in {file_name}")

            # --- normalize angle ---
            a_min, a_max = np.min(angles), np.max(angles)
            if a_max != a_min:
                angle_norm = (angles - a_min) / (a_max - a_min)
            else:
                angle_norm = np.zeros_like(angles)
                print(f"[WARN] Constant angle in {file_name}")

            # combine normalized features
            processed_points = np.stack((dist_norm, angle_norm), axis=1).astype(np.float32)
            processed_data.append(processed_points)
            valid_labels.append(label)
            quality_scores.append(quality_score)
            file_list.append(file_name)

        except Exception as e:
            print(f"[ERROR] Processing {full_path}: {e}")

    processed_data = np.array(processed_data, dtype=object)  # frames may vary slightly in length
    labels_array = np.array(valid_labels, dtype=np.int64)
    quality_array = np.array(quality_scores, dtype=np.float32)

    return processed_data, labels_array, quality_array, file_list


if __name__ == "__main__":
    np.random.seed(42)
    data, labels, quality, files = getLidar()

    num_samples = len(data)
    indices = np.random.permutation(num_samples)

    train_end = int(0.6 * num_samples)
    val_end = int(0.8 * num_samples)

    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]

    train_data, val_data, test_data = data[train_idx], data[val_idx], data[test_idx]
    train_labels, val_labels, test_labels = labels[train_idx], labels[val_idx], labels[test_idx]

    print(f"[INFO] Training: {len(train_data)}")
    print(f"[INFO] Validation: {len(val_data)}")
    print(f"[INFO] Testing: {len(test_data)}")

    # Optional save
    # np.savez("<OUTPUT_DIR>/lidar/train.npz", data=train_data, label=train_labels)
    # np.savez("<OUTPUT_DIR>/lidar/val.npz", data=val_data, label=val_labels)
    # np.savez("<OUTPUT_DIR>/lidar/test.npz", data=test_data, label=test_labels)
