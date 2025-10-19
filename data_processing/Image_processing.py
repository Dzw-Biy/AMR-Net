import os
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
# from Quality_acess.image_quality import get_quality   # optional module


# Deepsense6G scenario-specific statistics
# Scenario 9 original: mean=[0.3845, 0.3968, 0.3751], std=[0.2925, 0.2884, 0.2434]
# Scenario 8 original: mean=[0.4183, 0.4267, 0.4412], std=[0.2938, 0.2623, 0.2287]
IMG_MEAN = (0.4183, 0.4267, 0.4412)
IMG_STD = (0.2938, 0.2623, 0.2287)
IMG_SIZE = (224, 224)



def getImg():
    """
    Load RGB images, resize to (224,224), normalize, and compute quality scores.

    Returns:
        processed_images : np.ndarray [N,3,224,224]
        labels_array     : np.ndarray [N,]
        quality_array    : np.ndarray [N,]
        file_list        : list[str]
    """
    img_resize = transforms.Resize(IMG_SIZE)
    img_norm = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    proc_pipe = transforms.Compose([
        transforms.ToPILImage(),
        img_resize,
        transforms.ToTensor(),
        img_norm
    ])

    # ---- Dataset configuration ----
    csv_file = "<DATA_DIR>/scenario8.csv"
    base_dir = "<DATA_DIR>/scenario8/img_weather/night/"

    df = pd.read_csv(csv_file)
    img_paths = df["unit1_rgb"].tolist()
    labels = df["unit1_beam_index"].tolist()

    processed_images, valid_labels, quality_scores, file_list = [], [], [], []

    # ---- Iterate through all images ----
    for path, label in zip(img_paths, labels):
        file_name = os.path.basename(path)
        full_path = os.path.join(base_dir, path.lstrip("./"))

        if not os.path.exists(full_path):
            print(f"[WARN] Missing file: {full_path}")
            continue

        try:
            img = Image.open(full_path).convert("RGB")

            # Image quality score (if available)
            try:
                from Quality_acess.image_quality import get_quality
                quality_score = get_quality(img)
            except Exception:
                quality_score = 0.8  # default fallback

            # Process to normalized tensor
            img_np = np.array(img)
            img_proc = proc_pipe(img_np)
            processed_images.append(img_proc.numpy())
            valid_labels.append(label)
            quality_scores.append(quality_score)
            file_list.append(file_name)

        except Exception as e:
            print(f"[ERROR] Processing {full_path}: {e}")
            continue

    # ---- Final conversion ----
    processed_images = np.array(processed_images, dtype=np.float32)
    labels_array = np.array(valid_labels, dtype=np.int64)
    quality_array = np.array(quality_scores, dtype=np.float32)

    return processed_images, labels_array, quality_array, file_list


if __name__ == "__main__":
    np.random.seed(42)
    data, labels, quality, files = getImg()

    num_samples = data.shape[0]
    indices = np.random.permutation(num_samples)
    train_end = int(0.6 * num_samples)
    val_end = int(0.8 * num_samples)

    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    train_data, val_data, test_data = data[train_idx], data[val_idx], data[test_idx]
    train_labels, val_labels, test_labels = labels[train_idx], labels[val_idx], labels[test_idx]

    print(f"[INFO] Train set: {train_data.shape[0]}")
    print(f"[INFO] Val set:   {val_data.shape[0]}")
    print(f"[INFO] Test set:  {test_data.shape[0]}")

    # Optional save (disabled for privacy)
    # np.savez("<OUTPUT_DIR>/rgb/train.npz", data=train_data, label=train_labels)
    # np.savez("<OUTPUT_DIR>/rgb/val.npz", data=val_data, label=val_labels)
    # np.savez("<OUTPUT_DIR>/rgb/test.npz", data=test_data, label=test_labels)
