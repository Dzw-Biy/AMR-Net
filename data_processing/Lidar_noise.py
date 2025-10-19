import os
import numpy as np


MISSING_PROB = 0.50                    # Probability of entire frame loss
SNR_LEVELS_DB = [15, 5, -5, -15]       # Target SNR levels in dB
COMPUTE_CHAMFER = False                # Whether to compute Chamfer Distance
EPS = 1e-12


def pol2cart(r, theta):
    """Convert polar (r,theta) → Cartesian (x,y)."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def cart2pol(x, y):
    """Convert Cartesian (x,y) → polar (r,theta)."""
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    return r, th


def snr_db_cartesian(clean_xy, noisy_xy):
    """Compute SNR(dB) = E||X||² / E||N||² in Cartesian domain."""
    x0, y0 = clean_xy[:, 0], clean_xy[:, 1]
    x1, y1 = noisy_xy[:, 0], noisy_xy[:, 1]
    sig_p = np.mean(x0 * x0 + y0 * y0) + EPS
    noise_p = np.mean((x1 - x0) ** 2 + (y1 - y0) ** 2) + EPS
    return 10.0 * np.log10(sig_p / noise_p)


def chamfer_distance_xy(A, B):
    """Compute bidirectional Chamfer distance between two point sets (x,y)."""
    if not COMPUTE_CHAMFER:
        return np.nan
    from scipy.spatial import cKDTree
    if len(A) == 0 or len(B) == 0:
        return np.inf
    kda, kdb = cKDTree(A), cKDTree(B)
    d_ab, _ = kda.query(B, k=1)
    d_ba, _ = kdb.query(A, k=1)
    return float(np.mean(d_ab ** 2) + np.mean(d_ba ** 2))


def minmax01_per_column(arr2):
    """Normalize each column of (T,2) to [0,1]; constant column → 0."""
    out = np.zeros_like(arr2, dtype=np.float32)
    for j in range(2):
        col = arr2[:, j]
        cmin, cmax = np.min(col), np.max(col)
        out[:, j] = (col - cmin) / (cmax - cmin) if cmax > cmin else 0.0
    return out


def batch_minmax01(arr):
    """Apply min-max normalization for a batch (N,T,2)."""
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = minmax01_per_column(arr[i])
    return out


def save_npz_numeric(path, data, labels, extra=None):
    """Save numeric dataset to compressed .npz file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if extra is None:
        np.savez_compressed(path, data=data, label=labels)
    else:
        np.savez_compressed(path, data=data, label=labels, **extra)


def apply_missing_whole_sample_fixed(data_fixed, prob, seed):
    """Randomly zero-out entire frames with given probability."""
    rng = np.random.default_rng(seed)
    mask = rng.random(data_fixed.shape[0]) < prob
    out = data_fixed.copy()
    out[mask] = 0.0
    return out, mask


def add_cartesian_awgn_to_target_snr_frame(rt_T2, target_snr_db, rng):
    """
    Add AWGN noise to one LiDAR frame in (x,y) domain with target SNR(dB).
    Input:
        rt_T2: (T,2) array [r (m), θ (rad)]
    Output:
        noisy_rt: noisy (T,2)
        actual: actual SNR achieved
    """
    r = rt_T2[:, 0].astype(np.float32)
    th = rt_T2[:, 1].astype(np.float32)
    x, y = pol2cart(r, th)
    xy = np.stack([x, y], axis=1).astype(np.float32)

    sig_p = np.mean(xy[:, 0] ** 2 + xy[:, 1] ** 2) + EPS
    sigma2 = sig_p / (2.0 * (10.0 ** (target_snr_db / 10.0)))
    sigma = np.sqrt(sigma2).astype(np.float32)

    noise = rng.normal(0.0, sigma, size=xy.shape).astype(np.float32)
    xy_noisy = xy + noise
    actual = snr_db_cartesian(xy, xy_noisy)

    r_n, th_n = cart2pol(xy_noisy[:, 0], xy_noisy[:, 1])
    noisy_rt = np.stack([r_n.astype(np.float32), th_n.astype(np.float32)], axis=1)
    return noisy_rt, actual


def add_noise_set_target_snr_fixed(data_fixed, snr_db_target, seed):
    """
    Add AWGN with target SNR to a batch of LiDAR frames (N,T,2).
    Returns noisy data, actual SNRs, and optionally Chamfer distances.
    """
    rng = np.random.default_rng(seed)
    N, T, _ = data_fixed.shape
    out = np.empty_like(data_fixed, dtype=np.float32)
    snrs = np.zeros((N,), dtype=np.float32)
    cds = np.zeros((N,), dtype=np.float32) if COMPUTE_CHAMFER else None

    for i in range(N):
        noisy_rt, actual = add_cartesian_awgn_to_target_snr_frame(data_fixed[i], snr_db_target, rng)
        out[i] = noisy_rt
        snrs[i] = actual
        if COMPUTE_CHAMFER:
            x0, y0 = pol2cart(data_fixed[i, :, 0], data_fixed[i, :, 1])
            x1, y1 = pol2cart(noisy_rt[:, 0], noisy_rt[:, 1])
            cds[i] = chamfer_distance_xy(np.stack([x0, y0], 1), np.stack([x1, y1], 1))
    return out, snrs, cds


def lidar_processing(data_list, labels, tr_idx, va_idx, te_idx, OUT_DIR, RNG_SEED):
    """
    Process LiDAR dataset with multiple degradation modes and save to npz.

    Args:
        data_list : list of frames, each (T,2)
        labels    : list or np.ndarray of labels
        tr_idx, va_idx, te_idx : train/val/test split indices
        OUT_DIR   : output base directory
        RNG_SEED  : random seed
    """
    np.random.seed(RNG_SEED)

    # 1) Verify consistency and stack frames (N,T,2)
    T_set = {arr.shape[0] for arr in data_list}
    if len(T_set) != 1:
        raise ValueError(f"Inconsistent point count per frame: {sorted(list(T_set))}")
    T = list(T_set)[0]
    data_all = np.stack([np.asarray(a, dtype=np.float32).reshape(T, 2) for a in data_list], axis=0)
    labels = np.asarray(labels, dtype=np.int64)

    # 2) Split
    train_rt, val_rt, test_rt = data_all[tr_idx], data_all[va_idx], data_all[te_idx]
    train_labels, val_labels, test_labels = labels[tr_idx], labels[va_idx], labels[te_idx]

    # 3) Clean train set
    train_01 = batch_minmax01(train_rt)
    save_npz_numeric(os.path.join(OUT_DIR, "p50", "lidar", "train.npz"),
                     data=train_01, labels=train_labels)
    print(f"[SAVE] train_clean -> {train_01.shape[0]} samples (T={T})")

    # 4) Case1: Frame missing simulation
    val_miss, val_mask = apply_missing_whole_sample_fixed(val_rt, prob=MISSING_PROB, seed=RNG_SEED + 1)
    test_miss, test_mask = apply_missing_whole_sample_fixed(test_rt, prob=MISSING_PROB, seed=RNG_SEED + 2)
    val_miss_01, test_miss_01 = batch_minmax01(val_miss), batch_minmax01(test_miss)

    miss_dir = os.path.join(OUT_DIR, "p50", "lidar")
    save_npz_numeric(os.path.join(miss_dir, "val.npz"), data=val_miss_01, labels=val_labels,
                     extra={"missing_mask": val_mask})
    save_npz_numeric(os.path.join(miss_dir, "test.npz"), data=test_miss_01, labels=test_labels,
                     extra={"missing_mask": test_mask})
    print(f"[SAVE] case1 missing (p={MISSING_PROB:.2f}) -> val:{val_miss_01.shape[0]} test:{test_miss_01.shape[0]}")

    # 5) Case2–5: AWGN at target SNRs
    for k, snr_db in enumerate(SNR_LEVELS_DB, start=2):
        val_noisy, val_snrs, val_cds = add_noise_set_target_snr_fixed(val_rt, snr_db, seed=RNG_SEED + 3 + k)
        test_noisy, test_snrs, test_cds = add_noise_set_target_snr_fixed(test_rt, snr_db, seed=RNG_SEED + 13 + k)

        val_noisy_01, test_noisy_01 = batch_minmax01(val_noisy), batch_minmax01(test_noisy)
        print(f"[INFO] case{k}: target={snr_db:>4}dB | val avg={val_snrs.mean():.2f} | test avg={test_snrs.mean():.2f}")

        extra_val, extra_test = {"snr_db": val_snrs}, {"snr_db": test_snrs}
        if COMPUTE_CHAMFER:
            extra_val["cd_xy"], extra_test["cd_xy"] = val_cds, test_cds

        out_dir = os.path.join(OUT_DIR, f"snr{int(snr_db)}", "lidar")
        save_npz_numeric(os.path.join(out_dir, "val.npz"), data=val_noisy_01, labels=val_labels, extra=extra_val)
        save_npz_numeric(os.path.join(out_dir, "test.npz"), data=test_noisy_01, labels=test_labels, extra=extra_test)

    print(f"[DONE] train:{train_rt.shape[0]}  val:{val_rt.shape[0]}  test:{test_rt.shape[0]} (T={T})")
