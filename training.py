# train_picolt_vit_full.py
import json
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers #type:ignore

from picolt_vit_checker import build_picolt_vit, forward_lens

# -------------------------
# User paths / config
# -------------------------
DATASET_DIR = "picolt_dataset_enhanced"
H5_PATH = f"{DATASET_DIR}/dataset.h5"
CSV_PATH = f"{DATASET_DIR}/parameters.csv"
SPLITS_JSON = f"{DATASET_DIR}/splits.json"  
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

param_cols = [
    "lens_theta_E", "lens_e1", "lens_e2",
    "lens_center_x", "lens_center_y", "lens_gamma1", "lens_gamma2"
]

# -------------------------
# Utility: robust image normalization & squeeze
# -------------------------
def ensure_channel_and_normalize(arr):
    """
    Ensure arr has shape (N,H,W,1), squeeze trailing singleton dims,
    convert to float32 and normalize each image to [0,1].
    """
    # squeeze trailing singletons until channel dims ok
    if arr.ndim == 5 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    # if still 3D (N,H,W) add channel
    if arr.ndim == 3:
        arr = arr[..., None]
    # convert
    arr = arr.astype("float32")
    # per-image normalization
    # mins = arr.min(axis=(1,2,3), keepdims=True)
    # maxs = arr.max(axis=(1,2,3), keepdims=True)
    # arr = (arr - mins) / (maxs - mins + 1e-8)
    return arr

# -------------------------
# Load global parameters + splits.json
# -------------------------
print("Loading parameters CSV and splits JSON...")
params_df = pd.read_csv(CSV_PATH)
params_all = params_df[param_cols].to_numpy(dtype=np.float32)

with open(SPLITS_JSON, "r") as jf:
    splits_map = json.load(jf)   # expects keys 'train','val','test' each a list of indices

# Convert to numpy arrays for indexing
split_indices = {}
for key in ("train","val","test"):
    if key in splits_map:
        split_indices[key] = np.array(splits_map[key], dtype=np.int64)
    else:
        raise ValueError(f"Split '{key}' not present in {SPLITS_JSON}")

# -------------------------
# Robust loader for each split
# -------------------------
def load_split(split_name):
    """
    Loads lensed, source, params for given split (train/val/test).
    Uses indices from splits.json to slice from the full arrays.
    """
    assert split_name in ("train", "val", "test")

    with h5py.File(H5_PATH, "r") as f:
        lensed_all = f["lensed_images"][:]   # shape (N,128,128,...) or (N,128,128,1,1)
        source_all = f["source_images"][:]

    # fix dimensions and normalize
    lensed_all = ensure_channel_and_normalize(lensed_all)
    source_all = ensure_channel_and_normalize(source_all)

    # pick indices for this split
    idx = split_indices[split_name]
    lensed = lensed_all[idx]
    source = source_all[idx]
    params = params_all[idx]

    # sanity check
    if len(lensed) != len(params) or len(source) != len(params):
        raise ValueError(f"Length mismatch for {split_name}: "
                         f"{len(lensed)}/{len(source)} vs {len(params)}")

    return lensed, source, params

# -------------------------
# Load splits
# -------------------------
print("Loading dataset splits (this may take a moment)...")
lensed_train, source_train, params_train = load_split("train")
lensed_val,   source_val,   params_val   = load_split("val")
lensed_test,  source_test,  params_test  = load_split("test")

print("Shapes:")
print(" Train:", lensed_train.shape, params_train.shape, source_train.shape)
print(" Val:  ", lensed_val.shape, params_val.shape, source_val.shape)
print(" Test: ", lensed_test.shape, params_test.shape, source_test.shape)

# -------------------------
# Build TF datasets
# -------------------------
BATCH_SIZE = 32
train_ds = tf.data.Dataset.from_tensor_slices((lensed_train, source_train, params_train))
train_ds = train_ds.shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((lensed_val, source_val, params_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((lensed_test, source_test, params_test))
test_ds = test_ds.batch(BATCH_SIZE)

# -------------------------
# Build model + losses
# -------------------------
model = build_picolt_vit(image_size=128, param_dim=len(param_cols))
opt = optimizers.Adam(1e-4)

mse = keras.losses.MeanSquaredError()
mae = keras.losses.MeanAbsoluteError()
def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# -------------------------
# Training loop with dynamic λ
# -------------------------
EPOCHS = 35
lambda_param, lambda_src, lambda_phys = 1.0, 1.0, 2.0

history_losses = {"train": [], "val": []}
lambda_history = []

@tf.function
def train_step(lensed, source, params,
               lambda_param, lambda_src, lambda_phys):
    with tf.GradientTape() as tape:
        params_pred, source_pred = model(lensed, training=True)

        loss_param = mse(params, params_pred)

        loss_src_l1 = mae(source, source_pred)
        loss_src_ssim = ssim_loss(source, source_pred)
        loss_src = 0.5*loss_src_l1 + 0.5*loss_src_ssim

        lensed_recon = forward_lens(source_pred, params_pred)
        loss_phys = mse(lensed, lensed_recon)

        total = lambda_param*loss_param + lambda_src*loss_src + lambda_phys*loss_phys

    grads = tape.gradient(total, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss_param, loss_src, loss_phys, total

@tf.function
def val_step(lensed, source, params,
             lambda_param, lambda_src, lambda_phys):
    params_pred, source_pred = model(lensed, training=False)

    loss_param = mse(params, params_pred)
    loss_src_l1 = mae(source, source_pred)
    loss_src_ssim = ssim_loss(source, source_pred)
    loss_src = 0.5*loss_src_l1 + 0.5*loss_src_ssim

    lensed_recon = forward_lens(source_pred, params_pred)
    loss_phys = mse(lensed, lensed_recon)

    total = lambda_param*loss_param + lambda_src*loss_src + lambda_phys*loss_phys
    return loss_param, loss_src, loss_phys, total

print("\nStarting training loop")
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}------>")
    train_batches = 0
    train_losses = []
    for lensed_batch, source_batch, params_batch in train_ds:
        lp, ls, lphys, tot = train_step(lensed_batch, source_batch, params_batch,
                                        lambda_param, lambda_src, lambda_phys)
        train_losses.append([lp.numpy(), ls.numpy(), lphys.numpy(), tot.numpy()])
        train_batches += 1

    val_batches = 0
    val_losses = []
    for lensed_batch, source_batch, params_batch in val_ds:
        lp, ls, lphys, tot = val_step(lensed_batch, source_batch, params_batch,
                                      lambda_param, lambda_src, lambda_phys)
        val_losses.append([lp.numpy(), ls.numpy(), lphys.numpy(), tot.numpy()])
        val_batches += 1

    mean_train = np.mean(train_losses, axis=0)
    mean_val = np.mean(val_losses, axis=0)
    history_losses["train"].append(mean_train)
    history_losses["val"].append(mean_val)

    print(f" Train -> param={mean_train[0]:.4f}, src={mean_train[1]:.4f}, phys={mean_train[2]:.4f}, total={mean_train[3]:.4f}")
    print(f" Val   -> param={mean_val[0]:.4f}, src={mean_val[1]:.4f}, phys={mean_val[2]:.4f}, total={mean_val[3]:.4f}")

    # dynamic lambda update based on train losses
    inv_losses = np.array([1.0/(mean_train[0]+1e-8), 1.0/(mean_train[1]+1e-8), 1.0/(mean_train[2]+1e-8)])
    lambdas = inv_losses / np.sum(inv_losses) * 3.0
    lambda_param, lambda_src, lambda_phys = lambdas
    lambda_history.append([float(lambda_param), float(lambda_src), float(lambda_phys)])
    print(f" Updated lambdas -> param={lambda_param:.3f}, src={lambda_src:.3f}, phys={lambda_phys:.3f}")

# -------------------------
# Final test evaluation
# -------------------------
print("\nRunning final test evaluation...")
test_losses = []
for lensed_batch, source_batch, params_batch in test_ds:
    lp, ls, lphys, tot = val_step(lensed_batch, source_batch, params_batch,
                                  lambda_param, lambda_src, lambda_phys)
    test_losses.append([lp.numpy(), ls.numpy(), lphys.numpy(), tot.numpy()])
mean_test = np.mean(test_losses, axis=0)
print("Final Test Results -> param={:.4f}, src={:.4f}, phys={:.4f}, total={:.4f}".format(*mean_test))

# -------------------------
# Physics validation if available
# -------------------------
PHYSICS_IMG_H5 = os.path.join(DATASET_DIR, "physics_validation_image.h5")
PHYSICS_PARAMS_CSV = os.path.join(DATASET_DIR, "physics_validation_params.csv")

if os.path.exists(PHYSICS_IMG_H5) and os.path.exists(PHYSICS_PARAMS_CSV):
    print("Running physics validation set...")

    with h5py.File(PHYSICS_IMG_H5, "r") as f:
        lensed_phys = f["lensed_images"][:]
        source_phys = f["source_images"][:]

    lensed_phys = ensure_channel_and_normalize(lensed_phys)
    source_phys = ensure_channel_and_normalize(source_phys)

    params_phys = pd.read_csv(PHYSICS_PARAMS_CSV)[param_cols].to_numpy(dtype=np.float32)

    phys_ds = tf.data.Dataset.from_tensor_slices(
        (lensed_phys, source_phys, params_phys)
    ).batch(BATCH_SIZE)

    phys_losses = []
    for lensed_batch, source_batch, params_batch in phys_ds:
        lp, ls, lphys, tot = val_step(lensed_batch, source_batch, params_batch,
                                      lambda_param, lambda_src, lambda_phys)
        phys_losses.append(lphys.numpy())

    print("Physics Validation Loss:", np.mean(phys_losses))
else:
    print("Physics validation files not found; skipping.")


# -------------------------
# Save model + plots
# -------------------------
model.save(os.path.join(CHECKPOINT_DIR, "picolt_vit_model.keras"))
print("Saved model to", os.path.join(CHECKPOINT_DIR, "picolt_vit_model.keras"))

train_arr = np.array(history_losses["train"])
val_arr = np.array(history_losses["val"])
lambda_arr = np.array(lambda_history)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_arr[:,3], label="Train Total")
plt.plot(val_arr[:,3], label="Val Total")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Evolution of Total Loss"); plt.legend()

plt.subplot(1,2,2)
plt.plot(train_arr[:,0], label="Param Loss")
plt.plot(train_arr[:,1], label="Source Loss")
plt.plot(train_arr[:,2], label="Physics Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Component Losses (Train)"); plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_curves.png"))
plt.show()

plt.figure(figsize=(8,5))
plt.plot(lambda_arr[:,0], label="λ_param")
plt.plot(lambda_arr[:,1], label="λ_src")
plt.plot(lambda_arr[:,2], label="λ_phys")
plt.xlabel("Epoch"); plt.ylabel("Weight Value"); plt.title("Dynamic λ Evolution"); plt.legend()
plt.savefig(os.path.join(CHECKPOINT_DIR, "lambda_curves.png"))
plt.show()
