import os
import numpy as np


MISSING_PROB   = 0.50              
SNR_LEVELS_DB  = [15, 5, -5, -15]    
COMPUTE_CHAMFER = False             
EPS = 1e-12

# ===== 工具函数 =====
def pol2cart(r, theta):
    x = r * np.cos(theta); y = r * np.sin(theta)
    return x, y

def cart2pol(x, y):
    r = np.hypot(x, y); th = np.arctan2(y, x)
    return r, th

def snr_db_cartesian(clean_xy, noisy_xy):

    x0, y0 = clean_xy[:,0], clean_xy[:,1]
    x1, y1 = noisy_xy[:,0], noisy_xy[:,1]
    sig_p   = np.mean(x0*x0 + y0*y0) + EPS
    noise_p = np.mean((x1-x0)**2 + (y1-y0)**2) + EPS
    return 10.0 * np.log10(sig_p / noise_p)

def chamfer_distance_xy(A, B):
    if not COMPUTE_CHAMFER: return np.nan
    from scipy.spatial import cKDTree
    if len(A) == 0 or len(B) == 0: return np.inf
    kda, kdb = cKDTree(A), cKDTree(B)
    d_ab, _ = kda.query(B, k=1)
    d_ba, _ = kdb.query(A, k=1)
    return float(np.mean(d_ab**2) + np.mean(d_ba**2))

def minmax01_per_column(arr2):

    out = np.zeros_like(arr2, dtype=np.float32)
    for j in range(2):
        col = arr2[:, j]
        cmin, cmax = np.min(col), np.max(col)
        out[:, j] = (col - cmin) / (cmax - cmin) if cmax > cmin else 0.0
    return out

def batch_minmax01(arr):  # (N,T,2) -> (N,T,2)
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = minmax01_per_column(arr[i])
    return out

def save_npz_numeric(path, data, labels, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if extra is None:
        np.savez_compressed(path, data=data, label=labels)
    else:
        np.savez_compressed(path, data=data, label=labels, **extra)

def apply_missing_whole_sample_fixed(data_fixed, prob, seed):

    rng = np.random.default_rng(seed)
    mask = rng.random(data_fixed.shape[0]) < prob
    out = data_fixed.copy()
    out[mask] = 0.0
    return out, mask



def add_cartesian_awgn_to_target_snr_frame(rt_T2, target_snr_db, rng):
  
    r  = rt_T2[:,0].astype(np.float32)
    th = rt_T2[:,1].astype(np.float32)
    x, y = pol2cart(r, th)
    xy   = np.stack([x, y], axis=1).astype(np.float32)

    sig_p  = np.mean(xy[:,0]**2 + xy[:,1]**2) + EPS
    sigma2 = sig_p / (2.0 * (10.0 ** (target_snr_db / 10.0)))
    sigma  = np.sqrt(sigma2).astype(np.float32)

    noise    = rng.normal(0.0, sigma, size=xy.shape).astype(np.float32)
    xy_noisy = xy + noise
    actual   = snr_db_cartesian(xy, xy_noisy)

    r_n, th_n = cart2pol(xy_noisy[:,0], xy_noisy[:,1])
    noisy_rt  = np.stack([r_n.astype(np.float32), th_n.astype(np.float32)], axis=1)
    return noisy_rt, actual

def add_noise_set_target_snr_fixed(data_fixed, snr_db_target, seed):
 
    rng = np.random.default_rng(seed)
    N, T, _ = data_fixed.shape
    out  = np.empty_like(data_fixed, dtype=np.float32)
    snrs = np.zeros((N,), dtype=np.float32)
    cds  = np.zeros((N,), dtype=np.float32) if COMPUTE_CHAMFER else None

    for i in range(N):
        noisy_rt, actual = add_cartesian_awgn_to_target_snr_frame(data_fixed[i], snr_db_target, rng)
        out[i]  = noisy_rt
        snrs[i] = actual
        if COMPUTE_CHAMFER:
            x0, y0 = pol2cart(data_fixed[i,:,0], data_fixed[i,:,1])
            x1, y1 = pol2cart(noisy_rt[:,0],     noisy_rt[:,1])
            cds[i] = chamfer_distance_xy(np.stack([x0,y0],1), np.stack([x1,y1],1))
    return out, snrs, cds


def _se2_rigid_xy(xy, dx, dy, dpsi_rad):
    """对 (N,2) xy 施加 SE(2) 刚体变换"""
    c, s = np.cos(dpsi_rad), np.sin(dpsi_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy_new = (xy @ R.T) + np.array([dx, dy], dtype=np.float32)
    return xy_new

def _wrap_angle(theta):
    """wrap 到 (-pi, pi]"""
    out = (theta + np.pi) % (2*np.pi) - np.pi
    return out

def _rt_to_xy(rt_T2):
    r  = rt_T2[:,0].astype(np.float32)
    th = rt_T2[:,1].astype(np.float32)
    x = r * np.cos(th); y = r * np.sin(th)
    return np.stack([x, y], axis=1).astype(np.float32)

def _xy_to_rt(xy_T2):
    x, y = xy_T2[:,0].astype(np.float32), xy_T2[:,1].astype(np.float32)
    r  = np.hypot(x, y).astype(np.float32)
    th = np.arctan2(y, x).astype(np.float32)
    th = _wrap_angle(th)
    return np.stack([r, th], axis=1).astype(np.float32)

def _apply_sector_dropout(xy, rng, width_deg_range=(20.0, 60.0), drop_prob=1.0):
    """随机扇区丢失（模拟安装盲区/遮挡）；保持点数，用 r=0 表示缺失"""
    if rng.random() > drop_prob:
        return xy
    ang = np.arctan2(xy[:,1], xy[:,0])
    center = rng.uniform(-np.pi, np.pi)
    width  = np.deg2rad(rng.uniform(*width_deg_range))
    # 角度距离（考虑 wrap）
    diff = np.abs(_wrap_angle(ang - center))
    mask = diff <= (width / 2.0)
    xy2 = xy.copy()
    # 扇区内点清零（标记为缺失）
    xy2[mask] = 0.0
    return xy2

def _apply_forward_fov(xy, rng, fov_deg_range=(90.0, 160.0), enable_prob=0.6):
    """限制前向视场（例如车载雷达前向扇区）；超出 FOV 的点置零"""
    if rng.random() > enable_prob:
        return xy
    ang = np.arctan2(xy[:,1], xy[:,0])  # [-pi,pi]
    fov_deg = rng.uniform(*fov_deg_range)
    half = np.deg2rad(fov_deg / 2.0)
    # 定义车头朝 +x 方向，保留 |yaw|<=half 的点
    keep = (np.abs(ang) <= half)
    xy2 = xy.copy()
    xy2[~keep] = 0.0
    return xy2

def _apply_sparsify(xy, rng, keep_ratio_range=(0.7, 0.9)):
    """距离相关/随机稀疏：随机丢点(置零)，保持形状不变"""
    N = xy.shape[0]
    keep_ratio = rng.uniform(*keep_ratio_range)
    keep = rng.random(N) < keep_ratio
    xy2 = xy.copy()
    xy2[~keep] = 0.0
    return xy2

def _apply_metric_noise(xy, rng, rel_sigma_range=(0.02, 0.06), enable_prob=0.8):
    """测量噪声：在 xy 上加高斯噪声，方差相对场景尺度设置"""
    if rng.random() > enable_prob:
        return xy
    r = np.hypot(xy[:,0], xy[:,1])
    # 用非零半径的中位数作为尺度
    r_nonzero = r[r > 0]
    scale = (np.median(r_nonzero) if r_nonzero.size > 0 else np.mean(r)) + 1e-6
    sigma = rng.uniform(*rel_sigma_range) * float(scale)
    noise = rng.normal(0.0, sigma, size=xy.shape).astype(np.float32)
    return (xy + noise).astype(np.float32)

def _v2v_lidar_medium_one_frame(rt_T2, rng):
    """
    单帧 V2V 中档扰动：
      (1) 刚体位姿差异：dx,dy∈[-2,2]m，yaw∈[-15°,15°]
      (2) 前向FOV限制：90°–160°（概率触发）
      (3) 随机扇区缺失：20°–60° 扇区（概率触发）
      (4) 稀疏化：keep 0.7–0.9
      (5) 度量噪声：相对尺度 0.02–0.06
    """
    # 极->直角
    xy = _rt_to_xy(rt_T2)

    # (1) 刚体位姿（SE(2)）
    dx = rng.uniform(-2.0, 2.0)
    dy = rng.uniform(-2.0, 2.0)
    dpsi = np.deg2rad(rng.uniform(-15.0, 15.0))
    xy = _se2_rigid_xy(xy, dx, dy, dpsi)

    # (2) FOV 限制
    xy = _apply_forward_fov(xy, rng, fov_deg_range=(90.0, 160.0), enable_prob=0.6)

    # (3) 扇区缺失
    xy = _apply_sector_dropout(xy, rng, width_deg_range=(20.0, 60.0), drop_prob=0.7)

    # (4) 稀疏化（置零）
    xy = _apply_sparsify(xy, rng, keep_ratio_range=(0.7, 0.9))

    # (5) 度量噪声
    xy = _apply_metric_noise(xy, rng, rel_sigma_range=(0.02, 0.06), enable_prob=0.8)

    # 回到极坐标
    rt = _xy_to_rt(xy)
    return rt

def add_v2v_lidar_set_medium(data_fixed, seed):
    """
    data_fixed: (N,T,2) 的极坐标 [r, theta]，单位与原始一致
    返回同形状的 V2V-扰动结果（中档强度）
    """
    rng = np.random.default_rng(seed)
    N = data_fixed.shape[0]
    out = np.empty_like(data_fixed, dtype=np.float32)
    for i in range(N):
        out[i] = _v2v_lidar_medium_one_frame(data_fixed[i], rng)
    return out


# def lidar_processing(data_list, labels, tr_idx, va_idx, te_idx, OUT_DIR, RNG_SEED):
#     np.random.seed(RNG_SEED)

#     T_set = {arr.shape[0] for arr in data_list}
#     if len(T_set) != 1:
#         raise ValueError(f"每帧点数不一致：{sorted(list(T_set))}")
#     T = list(T_set)[0]
#     data_all = np.stack([np.asarray(a, dtype=np.float32).reshape(T,2) for a in data_list], axis=0)  # (N,T,2)
#     labels   = np.asarray(labels, dtype=np.int64)

#     # 2) 划分
#     train_rt = data_all[tr_idx]
#     val_rt   = data_all[va_idx]
#     test_rt  = data_all[te_idx]
#     train_labels = labels[tr_idx]
#     val_labels   = labels[va_idx]
#     test_labels  = labels[te_idx]

#     train_01 = batch_minmax01(train_rt)
#     save_npz_numeric(os.path.join(OUT_DIR, 'p50', 'lidar', 'train.npz'),
#                      data=train_01, labels=train_labels)
#     print(f"[SAVE] train_clean -> {train_01.shape[0]} samples, T={T}")

#     val_miss,  val_miss_mask  = apply_missing_whole_sample_fixed(val_rt,  prob=MISSING_PROB, seed=RNG_SEED+1)
#     test_miss, test_miss_mask = apply_missing_whole_sample_fixed(test_rt, prob=MISSING_PROB, seed=RNG_SEED+2)
#     val_miss_01  = batch_minmax01(val_miss)
#     test_miss_01 = batch_minmax01(test_miss)

#     miss_dir = os.path.join(OUT_DIR, "p50", 'lidar')
#     save_npz_numeric(os.path.join(miss_dir, 'val.npz'),
#                      data=val_miss_01, labels=val_labels, extra={"missing_mask": val_miss_mask})
#     save_npz_numeric(os.path.join(miss_dir, 'test.npz'),
#                      data=test_miss_01, labels=test_labels, extra={"missing_mask": test_miss_mask})
#     print(f"[SAVE] case1 missing (p={MISSING_PROB:.2f}) -> val:{val_miss_01.shape[0]} test:{test_miss_01.shape[0]}")

#     # 5) case2–5：按目标SNR在(x,y)域加噪（允许负SNR）
#     for k, snr_db in enumerate(SNR_LEVELS_DB, start=2):
#         val_noisy,  val_snrs,  val_cds  = add_noise_set_target_snr_fixed(val_rt,  snr_db, seed=RNG_SEED+3+k)
#         test_noisy, test_snrs, test_cds = add_noise_set_target_snr_fixed(test_rt, snr_db, seed=RNG_SEED+13+k)

#         val_noisy_01  = batch_minmax01(val_noisy)
#         test_noisy_01 = batch_minmax01(test_noisy)

#         print(f"[INFO] case{k} target SNR={snr_db:>4} dB | val avg={val_snrs.mean():.2f} dB | test avg={test_snrs.mean():.2f} dB")

#         extra_val  = {"snr_db": val_snrs}
#         extra_test = {"snr_db": test_snrs}
#         if COMPUTE_CHAMFER:
#             extra_val["cd_xy"]  = val_cds
#             extra_test["cd_xy"] = test_cds

#         out_dir = os.path.join(OUT_DIR, f"snr{int(snr_db)}", 'lidar')
#         save_npz_numeric(os.path.join(out_dir, 'val.npz'),  data=val_noisy_01,  labels=val_labels,  extra=extra_val)
#         save_npz_numeric(os.path.join(out_dir, 'test.npz'), data=test_noisy_01, labels=test_labels, extra=extra_test)

#     # 6) 摘要
#     print(f"[DONE] train:{train_rt.shape[0]}  val:{val_rt.shape[0]}  test:{test_rt.shape[0]}  (T={T})")


def lidar_processing(data_list, labels, tr_idx, va_idx, te_idx, OUT_DIR, RNG_SEED):
    np.random.seed(RNG_SEED)

    # 1) 组装为 (N,T,2)
    T_set = {arr.shape[0] for arr in data_list}
    if len(T_set) != 1:
        raise ValueError(f"每帧点数不一致：{sorted(list(T_set))}")
    T = list(T_set)[0]
    data_all = np.stack([np.asarray(a, dtype=np.float32).reshape(T,2) for a in data_list], axis=0)  # (N,T,2)
    labels   = np.asarray(labels, dtype=np.int64)

    # 2) 划分
    train_rt = data_all[tr_idx]
    val_rt   = data_all[va_idx]
    test_rt  = data_all[te_idx]
    train_labels = labels[tr_idx]
    val_labels   = labels[va_idx]
    test_labels  = labels[te_idx]

    # 3) 训练集：与原流程一致（做 0-1 归一化并保存）
    train_01 = batch_minmax01(train_rt)
    save_npz_numeric(os.path.join(OUT_DIR, 'v2v', 'lidar', 'train.npz'),
                     data=train_01, labels=train_labels)
    print(f"[SAVE] train_clean -> {train_01.shape[0]} samples, T={T}")

    # # 4) case1：整帧缺失（保持原实现与输出路径）
    # val_miss,  val_miss_mask  = apply_missing_whole_sample_fixed(val_rt,  prob=MISSING_PROB, seed=RNG_SEED+1)
    # test_miss, test_miss_mask = apply_missing_whole_sample_fixed(test_rt, prob=MISSING_PROB, seed=RNG_SEED+2)
    # val_miss_01  = batch_minmax01(val_miss)
    # test_miss_01 = batch_minmax01(test_miss)

    # miss_dir = os.path.join(OUT_DIR, "p50", 'lidar')
    # save_npz_numeric(os.path.join(miss_dir, 'val.npz'),
    #                  data=val_miss_01, labels=val_labels, extra={"missing_mask": val_miss_mask})
    # save_npz_numeric(os.path.join(miss_dir, 'test.npz'),
    #                  data=test_miss_01, labels=test_labels, extra={"missing_mask": test_miss_mask})
    # print(f"[SAVE] case1 missing (p={MISSING_PROB:.2f}) -> val:{val_miss_01.shape[0]} test:{test_miss_01.shape[0]}")

    # 5) case2：V2V 中档扰动（取代原 SNR 循环；只生成一个档次，文件名固定为 val.npz/test.npz）
    v2v_dir = os.path.join(OUT_DIR, "v2v", "lidar")
    val_v2v  = add_v2v_lidar_set_medium(val_rt,  seed=RNG_SEED+101)
    test_v2v = add_v2v_lidar_set_medium(test_rt, seed=RNG_SEED+202)
    val_v2v_01  = batch_minmax01(val_v2v)
    test_v2v_01 = batch_minmax01(test_v2v)

    save_npz_numeric(os.path.join(v2v_dir, 'val.npz'),  data=val_v2v_01,  labels=val_labels)
    save_npz_numeric(os.path.join(v2v_dir, 'test.npz'), data=test_v2v_01, labels=test_labels)
    print(f"[SAVE] case2 V2V (medium) -> val:{val_v2v_01.shape[0]} test:{test_v2v_01.shape[0]}")

    # 6) 摘要
    print(f"[DONE] train:{train_rt.shape[0]}  val:{val_rt.shape[0]}  test:{test_rt.shape[0]}  (T={T})")