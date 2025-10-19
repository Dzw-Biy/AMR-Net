import os
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
# from Quality_acess.image_quality import get_quality   # Optional external quality module


def getGray():
    """
    Load grayscale images, resize to (224,224), normalize to [-1,1],
    and compute corresponding quality scores.

    Returns:
        processed_images : np.ndarray [N,1,224,224]
        labels_array     : np.ndarray [N,]
        quality_array    : np.ndarray [N,]
        file_list        : list[str]
    """

    # --- Image preprocessing pipeline ---
    img_resize = transforms.Resize((224, 224))
    img_norm = transforms.Normalize(mean=(0.5,), std=(0.5,))
    proc_pipe = transforms.Compose([
        img_resize,
        transforms.ToTensor(),
        img_norm
    ])

    # --- Dataset sources ---
    csv_file = "<DATA_DIR>/scenario8.csv"
    base_dir = "<DATA_DIR>/scenario8"

    df = pd.read_csv(csv_file)
    img_paths = df["unit1_rgb"].tolist()
    labels = df["unit1_beam_index"].tolist()

    # --- Storage containers ---
    processed_images = []
    valid_labels = []
    quality_scores = []
    file_list = []

    # --- Iterate through image list ---
    for path, label in zip(img_paths, labels):
        file_name = os.path.basename(path)
        full_path = os.path.join(base_dir, path.lstrip("./"))

        if not os.path.exists(full_path):
            print(f"[WARN] File not found: {full_path}")
            continue

        try:
            # Gray + RGB copies (gray for model, RGB for quality eval)
            img_gray = Image.open(full_path).convert('L')
            img_rgb = Image.open(full_path).convert('RGB')

            # Quality metric (placeholder if unavailable)
            try:
                from Quality_acess.image_quality import get_quality
                quality_score = get_quality(img_rgb)
            except Exception:
                quality_score = 0.8  # default if module missing

            img_proc = proc_pipe(img_gray)  # -> tensor [1,H,W]
            processed_images.append(img_proc.numpy())
            valid_labels.append(label)
            quality_scores.append(quality_score)
            file_list.append(file_name)

        except Exception as e:
            print(f"[ERROR] Processing {full_path}: {e}")
            continue

    # --- Final conversion ---
    processed_images = np.array(processed_images, dtype=np.float32)
    labels_array = np.array(valid_labels, dtype=np.int64)
    quality_array = np.array(quality_scores, dtype=np.float32)

    return processed_images, labels_array, quality_array, file_list


if __name__ == '__main__':
    np.random.seed(42)
    data, labels, quality, files = getGray()

    num_samples = data.shape[0]
    indices = np.random.permutation(num_samples)

    train_end = int(0.6 * num_samples)
    val_end = int(0.8 * num_samples)

    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]

    train_data, val_data, test_data = data[train_idx], data[val_idx], data[test_idx]
    train_labels, val_labels, test_labels = labels[train_idx], labels[val_idx], labels[test_idx]

    print(f"[INFO] Training set: {train_data.shape[0]}")
    print(f"[INFO] Validation set: {val_data.shape[0]}")
    print(f"[INFO] Test set: {test_data.shape[0]}")

    # Optional saving (disabled for privacy)
    # np.savez("<OUTPUT_DIR>/gray/train.npz", data=train_data, label=train_labels)
    # np.savez("<OUTPUT_DIR>/gray/val.npz", data=val_data, label=val_labels)
    # np.savez("<OUTPUT_DIR>/gray/test.npz", data=test_data, label=test_labels)
