import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


IMG_SIZE = (224, 224)
MISSING_PROB = 0.50      # Probability of entire image being zeroed out
PSNR_LEVELS = [-5, -15]  # Noise levels (in dB)



def to_tensor_01(img_pil_gray):
    """Convert a PIL grayscale image -> torch tensor [1,H,W] ∈ [0,1]."""
    pipe = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor()
    ])
    return pipe(img_pil_gray)


def normalize_neg1_1(t):
    """Map [0,1] → [-1,1]."""
    return (t - 0.5) / 0.5


def denormalize_neg1_1(t):
    """Map [-1,1] → [0,1]."""
    return t * 0.5 + 0.5


@torch.no_grad()
def psnr_db(ref01: torch.Tensor, est: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute PSNR (in dB) between reference and estimate images."""
    diff = (ref01 - est).to(torch.float64)
    mse = torch.mean(diff * diff).item()
    if mse <= 1e-20:
        return 99.0
    return 10.0 * np.log10((max_val ** 2) / mse)


@torch.no_grad()
def add_awgn_to_target_psnr(img01: torch.Tensor,
                            target_psnr_db: float,
                            metric: str = "preclamp",
                            clip_output: bool = True,
                            rng=None):
    """
    Add AWGN to an image with a target PSNR (dB).

    Args:
        img01: clean image tensor [1,H,W] ∈ [0,1]
        target_psnr_db: desired PSNR in dB
        metric: 'preclamp' (measure PSNR before clamping) or 'unclamped'
        clip_output: whether to clamp output into [0,1]
        rng: optional random generator

    Returns:
        noisy_out: noisy image tensor
        actual_psnr: actual PSNR achieved (float)
    """
    assert img01.dtype.is_floating_point, "img01 must be float tensor"

    # PSNR = 10 * log10(1 / MSE)
    target_mse = 10.0 ** (-target_psnr_db / 10.0)
    sigma = np.sqrt(target_mse)

    noise = torch.randn_like(img01, generator=rng) * sigma if rng else torch.randn_like(img01) * sigma
    noisy_raw = img01 + noise

    actual_psnr = psnr_db(img01, noisy_raw, max_val=1.0)

    if metric == "preclamp":
        noisy_out = torch.clamp(noisy_raw, 0.0, 1.0) if clip_output else noisy_raw
    elif metric == "unclamped":
        noisy_out = noisy_raw
    else:
        raise ValueError("metric must be 'preclamp' or 'unclamped'.")

    return noisy_out, actual_psnr



def load_gray_dataset(csv_file: str, base_dir: str):
    """
    Load grayscale images and labels from dataset CSV.

    Returns:
        data_01: np.ndarray [N,1,H,W] in [0,1]
        labels_arr: np.ndarray [N,]
        files: list of filenames
    """
    df = pd.read_csv(csv_file)
    img_paths = df["unit1_rgb"].tolist()
    labels = df["unit1_beam_index"].tolist()

    data_01, labels_arr, files = [], [], []

    for path, label in zip(img_paths, labels):
        file_name = os.path.basename(path)
        full_path = os.path.join(base_dir, path.lstrip("./"))
        if not os.path.exists(full_path):
            print(f"[WARN] Not found: {full_path}")
            continue
        try:
            img = Image.open(full_path).convert('L')
            t01 = to_tensor_01(img)
            data_01.append(t01.numpy())
            labels_arr.append(label)
            files.append(file_name)
        except Exception as e:
            print(f"[ERROR] Failed to load {full_path}: {e}")

    data_01 = np.array(data_01, dtype=np.float32)
    labels_arr = np.array(labels_arr, dtype=np.int64)
    return data_01, labels_arr, files



def split_indices(n, seed, train_ratio=0.6, val_ratio=0.2):
    """Generate train/val/test indices with a fixed seed."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def apply_missing_whole_sample(data01, prob, seed):
    """
    Randomly zero out entire samples to simulate sensor failure or missing data.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(data01.shape[0]) < prob
    out = data01.copy()
    out[mask] = 0.0
    return out, mask


def add_noise_set_to_psnr(data01, target_psnr_db):
    """
    Add AWGN to each sample in a dataset to achieve a target PSNR.
    Returns:
        noisy [N,1,H,W], psnrs [N]
    """
    noisy = np.empty_like(data01)
    psnrs = []
    for i in range(data01.shape[0]):
        clean_t = torch.from_numpy(data01[i])
        noisy_t, act = add_awgn_to_target_psnr(clean_t, target_psnr_db)
        noisy[i] = noisy_t.numpy()
        psnrs.append(act)
    return noisy, np.array(psnrs, dtype=np.float32)


def save_npz(path, data_neg1_1, labels):
    """Save normalized gray dataset to .npz file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, data=data_neg1_1, label=labels)



def gray_processing(data01, labels, tr_idx, va_idx, te_idx, OUT_DIR, RNG_SEED):
    """
    Perform full gray image preprocessing and augmentation pipeline.
    - Normalize training data
    - Simulate missing data
    - Add controlled noise levels for validation/test

    Args:
        data01: np.ndarray [N,1,H,W] ∈ [0,1]
        labels: np.ndarray [N,]
        tr_idx, va_idx, te_idx: dataset splits
        OUT_DIR: output base directory
        RNG_SEED: random seed
    """
    np.random.seed(RNG_SEED)

    train01, val01, test01 = data01[tr_idx], data01[va_idx], data01[te_idx]
    train_labels, val_labels, test_labels = labels[tr_idx], labels[va_idx], labels[te_idx]

    # 1. Save clean training set (normalized to [-1,1])
    train_norm = normalize_neg1_1(torch.from_numpy(train01)).numpy().astype(np.float32)
    save_npz(os.path.join(OUT_DIR, 'snr-5', 'gray', "train.npz"), train_norm, train_labels)
    print(f"[SAVE] Train set -> {train_norm.shape[0]} samples")

    # 2. Simulate missing samples for validation/test
    val_miss01, _ = apply_missing_whole_sample(val01, prob=MISSING_PROB, seed=RNG_SEED + 1)
    test_miss01, _ = apply_missing_whole_sample(test01, prob=MISSING_PROB, seed=RNG_SEED + 2)

    val_miss_norm = normalize_neg1_1(torch.from_numpy(val_miss01)).numpy().astype(np.float32)
    test_miss_norm = normalize_neg1_1(torch.from_numpy(test_miss01)).numpy().astype(np.float32)

    save_npz(os.path.join(OUT_DIR, 'p50', 'gray', "val.npz"), val_miss_norm, val_labels)
    save_npz(os.path.join(OUT_DIR, 'p50', 'gray', "test.npz"), test_miss_norm, test_labels)
    print(f"[SAVE] Val/Test missing data (p={MISSING_PROB:.2f})")

    # 3. Add Gaussian noise at target PSNRs
    for psnr in PSNR_LEVELS:
        val_noisy01, val_psnrs = add_noise_set_to_psnr(val01, psnr)
        test_noisy01, test_psnrs = add_noise_set_to_psnr(test01, psnr)

        print(f"[INFO] val PSNR={psnr}dB → avg {val_psnrs.mean():.2f}dB")
        print(f"[INFO] test PSNR={psnr}dB → avg {test_psnrs.mean():.2f}dB")

        val_noisy_norm = normalize_neg1_1(torch.from_numpy(val_noisy01)).numpy().astype(np.float32)
        test_noisy_norm = normalize_neg1_1(torch.from_numpy(test_noisy01)).numpy().astype(np.float32)

        tag = f"snr{int(psnr)}"
        save_npz(os.path.join(OUT_DIR, tag, 'gray', "val.npz"), val_noisy_norm, val_labels)
        save_npz(os.path.join(OUT_DIR, tag, 'gray', "test.npz"), test_noisy_norm, test_labels)


if __name__ == "__main__":
    # Example usage (paths masked)
    CSV_FILE = "<DATA_DIR>/scenario8.csv"
    BASE_DIR = "<DATA_DIR>/scenario8/"
    OUT_DIR = "<OUTPUT_DIR>/gray_noise"

    data01, labels, files = load_gray_dataset(CSV_FILE, BASE_DIR)
    tr_idx, va_idx, te_idx = split_indices(len(data01), seed=42)
    gray_processing(data01, labels, tr_idx, va_idx, te_idx, OUT_DIR, RNG_SEED=42)
