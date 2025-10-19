import os
import numpy as np
import pandas as pd
import utm
# from Quality_acess.coord_quality import calculate_gps_quality  # optional external module


def min_max(arr, ax=0):
    """Performs min-max normalization along given axis."""
    return (arr - arr.min(axis=ax)) / (arr.max(axis=ax) - arr.min(axis=ax))


def xy_from_latlong(lat_long):
    """
    Convert latitude-longitude pairs to Cartesian (UTM) coordinates.
    Input shape: (N, 2) with columns [lat, lon].
    Output shape: (N, 2) with columns [x, y].
    """
    x, y, *_ = utm.from_latlon(lat_long[:, 0], lat_long[:, 1])
    return np.stack((x, y), axis=1)


def compute_distance_and_angle(data_array, base_station):
    """
    Compute bearing angle (deg) and great-circle distance (m)
    from each point in `data_array` to the `base_station`.
    """
    R = 6371000  # Earth radius in meters
    base_lat_rad, base_lon_rad = np.radians(base_station)
    lat_rad = np.radians(data_array[:, 0])
    lon_rad = np.radians(data_array[:, 1])

    dlat = lat_rad - base_lat_rad
    dlon = lon_rad - base_lon_rad

    a = np.sin(dlat / 2)**2 + np.cos(base_lat_rad) * np.cos(lat_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    x = np.sin(dlon) * np.cos(lat_rad)
    y = np.cos(base_lat_rad) * np.sin(lat_rad) - np.sin(base_lat_rad) * np.cos(lat_rad) * np.cos(dlon)
    bearing = np.arctan2(x, y)
    bearing_deg = (np.degrees(bearing) + 360) % 360

    return np.column_stack((bearing_deg, distance))


def quantize_data(data, num_bins=200):
    """Uniformly quantize normalized data into discrete bins."""
    resolution = 1.0 / num_bins
    quantized = np.floor(data / resolution).astype(int)
    return np.clip(quantized, 0, num_bins - 1)



def normalize_pos(pos1, pos2, norm_type=1):
    """
    Normalize or transform GPS coordinates.

    Args:
        pos1 : ndarray (1, 2) — base station coordinates [lat, lon]
        pos2 : ndarray (N, 2) — user/device coordinates
        norm_type : int
            1: Min-max normalization (raw lat/lon)
            2: Min-max normalization with N/S & E/W flipping (transfer aware)
            3: Convert to Cartesian (UTM), then min-max
            4: Cartesian + rotation alignment (base-station-centered)
            5: Polar coordinates (distance + angle), min-max

    Returns:
        pos_norm : ndarray (N, 2) normalized coordinates
    """
    if norm_type == 1:
        return min_max(pos2)

    elif norm_type == 2:
        pos_norm = min_max(pos2)
        avg_pos = np.mean(pos2, axis=0)
        if pos1[0, 0] > avg_pos[0]:
            pos_norm[:, 0] = 1 - pos_norm[:, 0]
        if pos1[0, 1] > avg_pos[1]:
            pos_norm[:, 1] = 1 - pos_norm[:, 1]
        return pos_norm

    elif norm_type == 3:
        return min_max(xy_from_latlong(pos2))

    elif norm_type == 4:
        pos2_cart = xy_from_latlong(pos2)
        pos_bs_cart = xy_from_latlong(pos1)
        pos_diff = pos2_cart - pos_bs_cart

        dist = np.linalg.norm(pos_diff, axis=1)
        ang = np.arctan2(pos_diff[:, 1], pos_diff[:, 0])

        dist_norm = dist / max(dist)

        avg_pos = np.mean(pos_diff, axis=0)
        avg_ang = np.arctan2(avg_pos[1], avg_pos[0])
        ang2 = np.where(ang > 0, ang, ang + 2 * np.pi)
        avg_ang2 = avg_ang + 2 * np.pi if avg_ang < 0 else avg_ang

        offset = np.pi / 2 - avg_ang2
        ang_final = ang2 + offset
        ang_norm = ang_final / np.pi

        return np.stack((dist_norm, ang_norm), axis=1)

    elif norm_type == 5:
        data_proc = compute_distance_and_angle(pos2, pos1[0])
        return min_max(data_proc)

    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")



def getCoord():
    """
    Read GPS coordinate files and generate normalized dataset.

    Returns:
        final_data : ndarray (N, 2)
        labels     : ndarray (N,)
        quality    : ndarray (N,)
        filenames  : list[str]
    """
    base_station = np.array([33.41932083333333, -111.92902222222223])
    csv_file = "<DATA_DIR>/scenario8.csv"
    base_dir = "<DATA_DIR>/scenario8/"

    df = pd.read_csv(csv_file)
    gps_paths = df["unit2_loc_cal"].tolist()
    labels = df["unit1_beam_index"].tolist()

    vectors, file_names, quality = [], [], []

    for rel_path in gps_paths:
        file_name = os.path.basename(rel_path)
        full_path = os.path.join(base_dir, rel_path.lstrip("./"))

        try:
            with open(full_path, 'r') as f:
                lines = [float(x.strip()) for x in f.readlines()]
                if len(lines) < 2:
                    raise ValueError(f"{file_name}: not enough data lines")
                vectors.append(np.array(lines[:2]))
                file_names.append(file_name)
                quality.append(0.8)  # Placeholder; real metric from GPS quality if available
        except Exception as e:
            print(f"⚠️ Error reading {full_path}: {e}")
            continue

    data_array = np.vstack(vectors)
    n = data_array.shape[0]
    base_matrix = np.tile(base_station, (n, 1))
    labels_array = np.array(labels[:n])
    quality_array = np.array(quality[:n])

    normalized_data = normalize_pos(base_matrix, data_array, norm_type=1)
    # quantized_data = quantize_data(normalized_data, num_bins=200)

    return normalized_data, labels_array, quality_array, file_names


if __name__ == "__main__":
    data, labels, quality, names = getCoord()

    np.random.seed(42)
    n = data.shape[0]
    idx = np.random.permutation(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_idx, val_idx, test_idx = idx[:train_end], idx[train_end:val_end], idx[val_end:]

    print(f"✅ Dataset loaded: {n} samples")
    print(f"Train / Val / Test: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")

    # Example saving (disabled for privacy)
    # np.savez("<OUTPUT_DIR>/coord/train.npz", data=data[train_idx], label=labels[train_idx])
