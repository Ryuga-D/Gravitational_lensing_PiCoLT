import os, numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from picolt_vit_checker import forward_lens, build_picolt_vit

DATASET_DIR = "picolt_dataset_enhanced"
PHYS_IMG_H5 = os.path.join(DATASET_DIR, "physics_validation_image.h5")
PHYS_PARAMS_CSV = os.path.join(DATASET_DIR, "physics_validation_params.csv")
MODEL_PATH = "checkpoints/picolt_vit_model.keras"   # update if needed

def ensure_channel_and_normalize(arr):
    # squeeze trailing singleton dims and add channel if missing
    if arr.ndim == 5 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim == 3:
        arr = arr[..., None]
    arr = arr.astype("float32")
    # per-image [0,1] normalization (same as training)
    mins = arr.min(axis=(1,2,3), keepdims=True)
    maxs = arr.max(axis=(1,2,3), keepdims=True)
    arr = (arr - mins) / (maxs - mins + 1e-8)
    return arr

# 1) Load physics files
if not (os.path.exists(PHYS_IMG_H5) and os.path.exists(PHYS_PARAMS_CSV)):
    raise FileNotFoundError("Physics files not found. Check paths.")

with h5py.File(PHYS_IMG_H5, "r") as f:
    lensed_phys = f["lensed_images"][:]   # shape (N,H,W) or (N,H,W,1)
    source_phys = f["source_images"][:]

lensed_phys = ensure_channel_and_normalize(lensed_phys)
source_phys = ensure_channel_and_normalize(source_phys)

params_phys_df = pd.read_csv(PHYS_PARAMS_CSV)
# ensure columns are in same order as model expects:
param_cols = ["lens_theta_E","lens_e1","lens_e2","lens_center_x","lens_center_y","lens_gamma1","lens_gamma2"]
if not all([c in params_phys_df.columns for c in param_cols]):
    raise ValueError("Physics params CSV missing expected columns or has different names.")
params_phys = params_phys_df[param_cols].to_numpy(dtype=np.float32)

print("Physics set shapes:", lensed_phys.shape, source_phys.shape, params_phys.shape)

# 2) Load model (weights) and build model
# If you saved model.keras, use tf.keras.models.load_model; else rebuild and load weights
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Loaded model from", MODEL_PATH)
except Exception as e:
    print("Could not load model directly, attempting to rebuild from builder.")
    model = build_picolt_vit(image_size=128, param_dim=len(param_cols))
    # If you saved weights separately, call model.load_weights(...)
    # model.load_weights("checkpoints/picolt_vit_weights.h5")

# 3) Compute per-sample physics loss using forward_lens
# We'll use forward_lens from picolt_vit_checker so it matches training
B = 64
n = lensed_phys.shape[0]
phys_losses = []
params_pred_all = []
src_pred_all = []

for i in range(0, n, B):
    batch_l = lensed_phys[i:i+B]
    # run model to predict params + source
    params_pred, src_pred = model.predict(batch_l, verbose=0)
    # ensure shapes: src_pred (B,H,W,1)
    lensed_recon = forward_lens(src_pred, params_pred)  # returns (B,H,W,1)
    # compute MSE per sample
    mse_per_sample = np.mean((batch_l - lensed_recon)**2, axis=(1,2,3))
    phys_losses.extend(mse_per_sample.tolist())
    params_pred_all.append(params_pred)
    src_pred_all.append(src_pred)

phys_losses = np.array(phys_losses)
print("Physics loss: mean=%.4f, median=%.4f, std=%.4f" % (phys_losses.mean(), np.median(phys_losses), phys_losses.std()))

# 4) Compare distributions (theta_E, magnification if available)
import matplotlib.pyplot as plt
# plot theta_E histograms: test vs physics (if you have test params saved)
# If you have params_test_df path, you can load and compare; otherwise just plot physics hist
plt.figure(figsize=(6,4))
plt.hist(params_phys_df["lens_theta_E"], bins=25, alpha=0.7)
plt.title("Physics validation theta_E distribution")
plt.xlabel("theta_E"); plt.ylabel("count")
plt.savefig("checkpoints/phys_thetaE_hist.png", dpi=150)
plt.close()

# 5) Show worst N cases (largest physics loss)
Nworst = 8
idx_sorted = np.argsort(phys_losses)[::-1]  # descending
worst_idx = idx_sorted[:Nworst]

# collect preds
params_pred_all = np.vstack(params_pred_all)
src_pred_all = np.vstack(src_pred_all)

# visualize grid: input lensed | re-lensed recon | predicted source | residual
import math
cols = 4
rows = Nworst
fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
for r, idx in enumerate(worst_idx):
    l_img = lensed_phys[idx][...,0]
    s_pred = src_pred_all[idx][...,0]
    p_pred = params_pred_all[idx:idx+1]
    re_l = forward_lens(src_pred_all[idx:idx+1], p_pred)[0,...,0]
    residual = l_img - re_l

    axs[r,0].imshow(l_img, cmap="viridis"); axs[r,0].set_title(f"Input lensed idx={idx}")
    axs[r,1].imshow(re_l, cmap="viridis"); axs[r,1].set_title("Re-lensed pred")
    axs[r,2].imshow(s_pred, cmap="viridis"); axs[r,2].set_title("Pred source")
    im = axs[r,3].imshow(residual, cmap="RdBu", vmin=-1, vmax=1); axs[r,3].set_title(f"Residual (MSE={phys_losses[idx]:.4f})")
    for c in range(cols):
        axs[r,c].axis("off")

plt.tight_layout()
plt.savefig("checkpoints/physics_worst_examples.png", dpi=150)
print("Saved worst-case visualizations to checkpoints/physics_worst_examples.png")
plt.close()

# 6) Save diagnostics CSV
diag_df = params_phys_df.copy()
diag_df["phys_mse"] = phys_losses
diag_df.to_csv("checkpoints/physics_validation_diagnostics.csv", index=False)
print("Saved diagnostics to checkpoints/physics_validation_diagnostics.csv")
