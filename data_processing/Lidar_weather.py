import os
import numpy as np
import pandas as pd
import scipy.io as sio


def limit_polar_range(polar_data, min_angle, max_angle):
    """Clip polar data to given angular range."""
    mask = (polar_data[:, 1] >= min_angle) & (polar_data[:, 1] <= max_angle)
    polar_data[~mask] = 0
    return polar_data


def load_lidar_data(file_path):
    """Load LiDAR point cloud (.mat) with 'data' key."""
    data = sio.loadmat(file_path)
    return np.array(data["data"], dtype=np.float32)


def load_angle_table(csv_path):
    """Load precomputed angle lookup table (CSV with 'angle_point' column)."""
    df = pd.read_csv(csv_path)
    return df["angle_point"].values.astype(np.float32)


def preprocess_lidar_data(lidar_data, angles):
    """
    Replace zero-angle points with known calibration angles.
    lidar_data: (N,2), angles: (N,)
    """
    lidar_copy = lidar_data.copy()
    zero_mask = lidar_copy[:, 1] == 0
    lidar_copy[zero_mask, 1] = angles[zero_mask]
    return lidar_copy


def simulate_rain(polar_data, noise_level=1.0, drop_prob=0.5):
    """Rain effect: random noise + random dropout."""
    distances, angles = polar_data[:, 0], polar_data[:, 1]
    noise_mask = np.random.rand(distances.shape[0]) > 0.6
    noise = noise_level * np.random.randn(*distances.shape)
    distances[noise_mask] += noise[noise_mask]

    drop_mask = np.random.rand(distances.shape[0]) > drop_prob
    distances = np.maximum(distances, 0)
    distances[~drop_mask] = 0
    angles[distances == 0] = 0
    return np.column_stack((distances, angles)).astype(np.float32)


def simulate_fog(polar_data, noise_level=1.0, max_range=10.0):
    """Fog effect: additive noise + range-based dropout."""
    distances, angles = polar_data[:, 0], polar_data[:, 1]
    noise_mask = np.random.rand(distances.shape[0]) > 0.4
    noise = noise_level * np.random.randn(*distances.shape)
    distances[noise_mask] += noise[noise_mask]
    distances = np.maximum(distances, 0)

    prob = 1 - (distances / max_range)  # farther → more likely to vanish
    drop_mask = np.random.rand(len(distances)) < prob
    distances[~drop_mask] = 0
    angles[distances == 0] = 0
    return np.column_stack((distances, angles)).astype(np.float32)


def simulate_snow(polar_data, noise_level=1.0, drop_prob=0.1):
    """Snow effect: mild Gaussian noise + partial occlusion."""
    distances, angles = polar_data[:, 0], polar_data[:, 1]
    noise_mask = np.random.rand(distances.shape[0]) > 0.8
    noise = noise_level * np.random.randn(*distances.shape) + 1.0
    distances[noise_mask] += noise[noise_mask]

    drop_mask = np.random.rand(distances.shape[0]) > drop_prob
    distances = np.maximum(distances, 0)
    distances[~drop_mask] = 0
    angles[distances == 0] = 0
    return np.column_stack((distances, angles)).astype(np.float32)


def process_all_mat_files(input_dir, output_dir, angle_csv):
    """
    Process all LiDAR .mat files in input_dir and generate synthetic
    rain/fog/snow versions into output_dir.
    """
    angles = load_angle_table(angle_csv)
    os.makedirs(output_dir, exist_ok=True)

    weather_modes = {
        "rainy": simulate_rain,
        "foggy": simulate_fog,
        "snowy": simulate_snow,
    }

    mat_files = [f for f in os.listdir(input_dir) if f.endswith(".mat")]
    print(f"[INFO] Found {len(mat_files)} LiDAR files in {input_dir}")

    for filename in mat_files:
        file_path = os.path.join(input_dir, filename)
        try:
            lidar_data = load_lidar_data(file_path)
            lidar_data = preprocess_lidar_data(lidar_data, angles)

            for weather, sim_func in weather_modes.items():
                weather_pc = sim_func(np.copy(lidar_data))
                save_path = os.path.join(
                    output_dir, weather, "unit1", "lidar_SCR_data", filename
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                sio.savemat(save_path, {"data": weather_pc})

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

    print(f"[DONE] All synthetic weather LiDAR datasets saved to {output_dir}")


if __name__ == "__main__":
    input_dir = "<DATA_DIR>/scenario8/unit1/lidar_SCR_data"
    output_dir = "<DATA_DIR>/scenario8/Lidar_weather"
    angle_csv = "<DATA_DIR>/scenario8/resources/angle_point.csv"

    process_all_mat_files(input_dir, output_dir, angle_csv)
