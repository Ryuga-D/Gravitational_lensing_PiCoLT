import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type:ignore
import h5py, pandas as pd, numpy as np

# ==============================
# 1. Dataset Loader
# ==============================
DATASET_DIR = "picolt_dataset_enhanced"
H5_PATH = f"{DATASET_DIR}/dataset.h5"
CSV_PATH = f"{DATASET_DIR}/parameters.csv"

# Load images
with h5py.File(H5_PATH, "r") as f:
    lensed_images = f["lensed_images"][:]     # maybe (N,128,128,1,1)
    source_images = f["source_images"][:]

# ---- Fix dimensions ----
if lensed_images.ndim == 5 and lensed_images.shape[-1] == 1:
    lensed_images = np.squeeze(lensed_images, axis=-1)  # -> (N,128,128,1)
if source_images.ndim == 5 and source_images.shape[-1] == 1:
    source_images = np.squeeze(source_images, axis=-1)

# ---- Ensure float32 ----
lensed_images = lensed_images.astype("float32")
source_images = source_images.astype("float32")

# ---- Per-image normalization [0,1] ----
def normalize(imgs):
    imgs_min = imgs.min(axis=(1,2,3), keepdims=True)
    imgs_max = imgs.max(axis=(1,2,3), keepdims=True)
    return (imgs - imgs_min) / (imgs_max - imgs_min + 1e-8)

lensed_images = normalize(lensed_images)
source_images = normalize(source_images)

# Load parameters
params = pd.read_csv(CSV_PATH)
param_cols = [
    "lens_theta_E","lens_e1","lens_e2",
    "lens_center_x","lens_center_y","lens_gamma1","lens_gamma2"
]
param_vectors = params[param_cols].to_numpy(dtype=np.float32)

print("Loaded dataset:", lensed_images.shape, source_images.shape, param_vectors.shape)

# Small dataset slice for testing
train_lensed = lensed_images[:64]
train_source = source_images[:64]
train_params = param_vectors[:64]


# ==============================
# 2. Vision Transformer Encoder
# ==============================
class PatchEmbedding(keras.layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.proj = keras.layers.Conv2D(embed_dim, patch_size, strides=patch_size, padding="valid")

    def call(self, x):
        x = self.proj(x)   # (B, H/ps, W/ps, E)
        B = tf.shape(x)[0]
        H, W, E = x.shape[1], x.shape[2], x.shape[3]  # static shape
        x = tf.reshape(x, (B, H*W, E))
        return x

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dense(embed_dim)
        ])

    def call(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

def build_picolt_vit(image_size=128, patch_size=16, embed_dim=128, 
                     num_layers=4, num_heads=4, mlp_dim=256, param_dim=7):

    inp = keras.layers.Input((image_size, image_size, 1))

    # Patch embedding
    patches = PatchEmbedding(patch_size, embed_dim)(inp)

    # Compute grid size
    Hc, Wc = image_size // patch_size, image_size // patch_size
    N = Hc * Wc

    # Trainable tokens
    cls_token = self_cls = tf.Variable(tf.zeros((1, 1, embed_dim)), trainable=True, name="cls_token")
    pos_embed = tf.Variable(tf.random.normal((1, N+1, embed_dim), stddev=0.02), trainable=True, name="pos_embed")

    # Lambda layer ensures dynamic batch size works
    def add_cls_and_pos(patches):
        B = tf.shape(patches)[0]
        cls_tokens = tf.tile(cls_token, [B, 1, 1])
        x = tf.concat([cls_tokens, patches], axis=1)
        return x + pos_embed

    x = keras.layers.Lambda(add_cls_and_pos)(patches)

    # Transformer blocks
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, mlp_dim)(x)

    # --- Head 1: Parameters ---
    cls_out = keras.layers.LayerNormalization()(x[:, 0, :])
    param_out = keras.layers.Dense(param_dim, name="param_out")(cls_out)

    # --- Head 2: Source Reconstruction ---
    patch_feats = x[:, 1:, :]
    feat_map = keras.layers.Reshape((Hc, Wc, embed_dim))(patch_feats)

    y = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(feat_map)
    y = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(y)
    y = keras.layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(y)
    y = keras.layers.Conv2DTranspose(8, 3, strides=2, padding="same", activation="relu")(y)
    source_out = keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="source_out")(y)

    return keras.Model(inp, [param_out, source_out], name="PICoLT_ViT")



# ==============================
# 3. Differentiable Forward Lens
# (SIS+Shear approx)
# ==============================

def forward_lens(source, lens_params, pixel_scale=0.2):
    """
    source: (B,H,W,1) in [0,1]
    lens_params: (B,7): [theta_E, e1, e2, cx, cy, g1, g2]
    returns: lensed image (B,H,W,1)
    """
    source = tf.convert_to_tensor(source)
    lens_params = tf.convert_to_tensor(lens_params)

    B = tf.shape(source)[0]
    H = tf.shape(source)[1]
    W = tf.shape(source)[2]

    # split params
    theta_E, e1, e2, cx, cy, g1, g2 = [
        lens_params[:, i][:, None, None] for i in range(7)
    ]  # each (B,1,1)

    # coordinate grid (H,W)
    xs = tf.linspace(-tf.cast(W, tf.float32) * 0.5 * pixel_scale,
                      tf.cast(W, tf.float32) * 0.5 * pixel_scale,
                      W)
    ys = tf.linspace(-tf.cast(H, tf.float32) * 0.5 * pixel_scale,
                      tf.cast(H, tf.float32) * 0.5 * pixel_scale,
                      H)
    X, Y = tf.meshgrid(xs, ys)              # (H,W)
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)

    # batch them to (B,H,W)
    Xb = tf.tile(X[None, ...], [B, 1, 1])
    Yb = tf.tile(Y[None, ...], [B, 1, 1])

    # lens equation (SIS + external shear)
    Xc = Xb - cx
    Yc = Yb - cy
    r = tf.sqrt(Xc**2 + Yc**2 + 1e-8)

    ax = theta_E * (Xc / r) + g1 * Xc + g2 * Yc
    ay = theta_E * (Yc / r) + g2 * Xc - g1 * Yc

    beta_x = Xb - ax
    beta_y = Yb - ay

    # convert source-plane coords to pixel coords
    px = (beta_x / pixel_scale) + tf.cast(W, tf.float32) * 0.5
    py = (beta_y / pixel_scale) + tf.cast(H, tf.float32) * 0.5

    # -------- Bilinear sampling with explicit (b,y,x,c) indices --------
    def sample(img, px, py):
        # img: (B,H,W,1)
        B = tf.shape(img)[0]
        H = tf.shape(img)[1]
        W = tf.shape(img)[2]

        # neighbors
        x0 = tf.floor(px); y0 = tf.floor(py)
        x1 = x0 + 1.0;     y1 = y0 + 1.0

        # clip to valid range
        x0i = tf.clip_by_value(tf.cast(x0, tf.int32), 0, W - 1)
        x1i = tf.clip_by_value(tf.cast(x1, tf.int32), 0, W - 1)
        y0i = tf.clip_by_value(tf.cast(y0, tf.int32), 0, H - 1)
        y1i = tf.clip_by_value(tf.cast(y1, tf.int32), 0, H - 1)

        # build (b,y,x,c) indices
        b = tf.reshape(tf.range(B, dtype=tf.int32), (B, 1, 1))
        b = tf.tile(b, [1, H, W])  # (B,H,W)
        c = tf.zeros_like(b)       # channel=0

        def gather(ix, iy):
            idx = tf.stack([b, iy, ix, c], axis=-1)    # (B,H,W,4)
            return tf.gather_nd(img, idx)              # (B,H,W)

        Ia = gather(x0i, y0i)
        Ib = gather(x0i, y1i)
        Ic = gather(x1i, y0i)
        Id = gather(x1i, y1i)

        # bilinear weights
        wa = (x1 - px) * (y1 - py)
        wb = (x1 - px) * (py - y0)
        wc = (px - x0) * (y1 - py)
        wd = (px - x0) * (py - y0)

        # ensure float
        wa = tf.cast(wa, tf.float32)
        wb = tf.cast(wb, tf.float32)
        wc = tf.cast(wc, tf.float32)
        wd = tf.cast(wd, tf.float32)

        out = Ia * wa + Ib * wb + Ic * wc + Id * wd    # (B,H,W)
        return out[..., None]                           # (B,H,W,1)

    return sample(source, px, py)


# ==============================
# 4. Build model and test losses
# ==============================
if __name__ == "__main__":
    # Build model
    model = build_picolt_vit()
    params_pred, src_pred = model(train_lensed[:4])

    # Losses
    mse = keras.losses.MeanSquaredError()
    mae = keras.losses.MeanAbsoluteError()

    loss_param = mse(train_params[:4], params_pred)
    loss_src = mae(train_source[:4], src_pred)
    lensed_pred = forward_lens(src_pred, params_pred)
    loss_phys = mse(train_lensed[:4], lensed_pred)

    print("Parameter loss:", float(loss_param.numpy()))
    print("Source loss:", float(loss_src.numpy()))
    print("Physics loss:", float(loss_phys.numpy()))
