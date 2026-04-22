# picolt_vit_improved.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type:ignore
import numpy as np

# ==============================
# 1. Vision Transformer Encoder (same as before)
# ==============================
class PatchEmbedding(keras.layers.Layer):
    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.proj = keras.layers.Conv2D(embed_dim, patch_size, strides=patch_size, padding="valid")

    def call(self, x):
        x = self.proj(x)
        B = tf.shape(x)[0]
        H, W, E = x.shape[1], x.shape[2], x.shape[3]
        x = tf.reshape(x, (B, H*W, E))
        return x

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout)
        ])

    def call(self, x, training=False):
        x = x + self.attn(self.norm1(x), self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x

# ==============================
# 2. Improved Decoder Components
# ==============================
class ResidualBlock(layers.Layer):
    """Residual block with skip connection"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        
    def call(self, x, training=False):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = x + residual
        x = tf.nn.relu(x)
        return x

class AttentionGate(layers.Layer):
    """Attention gate for skip connections"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.W_g = layers.Conv2D(filters, 1, padding='same')
        self.W_x = layers.Conv2D(filters, 1, padding='same')
        self.psi = layers.Conv2D(1, 1, padding='same', activation='sigmoid')
        
    def call(self, g, x):
        """g: gating signal (decoder), x: skip connection from encoder"""
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = tf.nn.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UpsampleBlock(layers.Layer):
    """Upsampling block with skip connection and attention"""
    def __init__(self, filters, use_attention=True, **kwargs):
        super().__init__(**kwargs)
        self.upsample = layers.Conv2DTranspose(filters, 3, strides=2, padding='same')
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate(filters)
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.residual = ResidualBlock(filters)
        
    def call(self, x, skip=None, training=False):
        x = self.upsample(x)
        
        if skip is not None:
            if self.use_attention:
                skip = self.attention(x, skip)
            x = tf.concat([x, skip], axis=-1)
        
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.residual(x, training=training)
        return x

# ==============================
# 3. Build Improved PICoLT
# ==============================
def build_improved_picolt(image_size=128, patch_size=16, embed_dim=128,
                          num_layers=4, num_heads=4, mlp_dim=256, 
                          param_dim=7, dropout=0.1):
    """
    Improved PICoLT with:
    - Enhanced decoder with skip connections
    - Residual blocks for better gradient flow
    - Attention gates for feature selection
    - Deeper architecture for source reconstruction
    """
    
    inp = keras.layers.Input((image_size, image_size, 1))
    
    # ============ ENCODER ============
    # Initial conv for skip connections
    x_init = layers.Conv2D(32, 3, padding='same', activation='relu', name='init_conv')(inp)
    x_skip_1 = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu', name='skip_1')(x_init)
    x_skip_2 = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu', name='skip_2')(x_skip_1)
    x_skip_3 = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu', name='skip_3')(x_skip_2)
    
    # Patch embedding
    patches = PatchEmbedding(patch_size, embed_dim)(inp)
    
    Hc, Wc = image_size // patch_size, image_size // patch_size
    N = Hc * Wc
    
    # Trainable tokens
    cls_token = tf.Variable(tf.zeros((1, 1, embed_dim)), trainable=True, name="cls_token")
    pos_embed = tf.Variable(tf.random.normal((1, N+1, embed_dim), stddev=0.02), 
                           trainable=True, name="pos_embed")
    
    def add_cls_and_pos(patches):
        B = tf.shape(patches)[0]
        cls_tokens = tf.tile(cls_token, [B, 1, 1])
        x = tf.concat([cls_tokens, patches], axis=1)
        return x + pos_embed
    
    x = keras.layers.Lambda(add_cls_and_pos)(patches)
    
    # Transformer blocks
    for i in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)(x)
    
    x = layers.LayerNormalization()(x)
    
    # ============ PARAMETER HEAD ============
    cls_out = x[:, 0, :]
    param_out = layers.Dense(128, activation='relu')(cls_out)
    param_out = layers.Dropout(dropout)(param_out)
    param_out = layers.Dense(param_dim, name="param_out")(param_out)
    
    # ============ IMPROVED SOURCE DECODER ============
    # Extract patch features
    patch_feats = x[:, 1:, :]
    feat_map = keras.layers.Reshape((Hc, Wc, embed_dim))(patch_feats)
    
    # Bottleneck processing
    y = layers.Conv2D(512, 3, padding='same', activation='relu')(feat_map)
    y = ResidualBlock(512)(y)
    
    # Upsample 1: 8x8 -> 16x16
    y = UpsampleBlock(256, use_attention=True)(y, skip=x_skip_3)
    
    # Upsample 2: 16x16 -> 32x32
    y = UpsampleBlock(128, use_attention=True)(y, skip=x_skip_2)
    
    # Upsample 3: 32x32 -> 64x64
    y = UpsampleBlock(64, use_attention=True)(y, skip=x_skip_1)
    
    # Upsample 4: 64x64 -> 128x128
    y = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(y)
    y = ResidualBlock(32)(y)
    
    # Final refinement layers
    y = layers.Conv2D(16, 3, padding='same', activation='relu')(y)
    y = layers.Conv2D(8, 3, padding='same', activation='relu')(y)
    
    # Output
    source_out = layers.Conv2D(1, 1, padding='same', activation='sigmoid', 
                              name="source_out")(y)
    
    return keras.Model(inp, [param_out, source_out], name="PICoLT_Improved")

# ==============================
# 4. VGG-based Perceptual Loss
# ==============================
class PerceptualLoss:
    """Perceptual loss using VGG16 features"""
    def __init__(self):
        # Load VGG16 without top layers
        vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
        # Extract features from multiple layers
        layer_names = ['block1_conv2', 'block2_conv2', 'block3_conv3']
        outputs = [vgg.get_layer(name).output for name in layer_names]
        self.model = keras.Model(vgg.input, outputs)
        self.model.trainable = False
        
    def __call__(self, y_true, y_pred):
        # Convert grayscale to RGB for VGG
        y_true_rgb = tf.repeat(y_true, 3, axis=-1)
        y_pred_rgb = tf.repeat(y_pred, 3, axis=-1)
        
        # Get features
        true_features = self.model(y_true_rgb)
        pred_features = self.model(y_pred_rgb)
        
        # Compute loss at each layer
        loss = 0.0
        for true_f, pred_f in zip(true_features, pred_features):
            loss += tf.reduce_mean(tf.abs(true_f - pred_f))
        
        return loss / len(true_features)

# ==============================
# 5. Forward Lens (same as before)
# ==============================
def forward_lens(source, lens_params, pixel_scale=0.2):
    """Differentiable lensing operator"""
    source = tf.convert_to_tensor(source)
    lens_params = tf.convert_to_tensor(lens_params)

    B = tf.shape(source)[0]
    H = tf.shape(source)[1]
    W = tf.shape(source)[2]

    theta_E, e1, e2, cx, cy, g1, g2 = [
        lens_params[:, i][:, None, None] for i in range(7)
    ]

    xs = tf.linspace(-tf.cast(W, tf.float32) * 0.5 * pixel_scale,
                      tf.cast(W, tf.float32) * 0.5 * pixel_scale, W)
    ys = tf.linspace(-tf.cast(H, tf.float32) * 0.5 * pixel_scale,
                      tf.cast(H, tf.float32) * 0.5 * pixel_scale, H)
    X, Y = tf.meshgrid(xs, ys)
    X = tf.cast(X, tf.float32)
    Y = tf.cast(Y, tf.float32)

    Xb = tf.tile(X[None, ...], [B, 1, 1])
    Yb = tf.tile(Y[None, ...], [B, 1, 1])

    Xc = Xb - cx
    Yc = Yb - cy
    r = tf.sqrt(Xc**2 + Yc**2 + 1e-8)

    ax = theta_E * (Xc / r) + g1 * Xc + g2 * Yc
    ay = theta_E * (Yc / r) + g2 * Xc - g1 * Yc

    beta_x = Xb - ax
    beta_y = Yb - ay

    px = (beta_x / pixel_scale) + tf.cast(W, tf.float32) * 0.5
    py = (beta_y / pixel_scale) + tf.cast(H, tf.float32) * 0.5

    def sample(img, px, py):
        B = tf.shape(img)[0]
        H = tf.shape(img)[1]
        W = tf.shape(img)[2]

        x0 = tf.floor(px); y0 = tf.floor(py)
        x1 = x0 + 1.0;     y1 = y0 + 1.0

        x0i = tf.clip_by_value(tf.cast(x0, tf.int32), 0, W - 1)
        x1i = tf.clip_by_value(tf.cast(x1, tf.int32), 0, W - 1)
        y0i = tf.clip_by_value(tf.cast(y0, tf.int32), 0, H - 1)
        y1i = tf.clip_by_value(tf.cast(y1, tf.int32), 0, H - 1)

        b = tf.reshape(tf.range(B, dtype=tf.int32), (B, 1, 1))
        b = tf.tile(b, [1, H, W])
        c = tf.zeros_like(b)

        def gather(ix, iy):
            idx = tf.stack([b, iy, ix, c], axis=-1)
            return tf.gather_nd(img, idx)

        Ia = gather(x0i, y0i)
        Ib = gather(x0i, y1i)
        Ic = gather(x1i, y0i)
        Id = gather(x1i, y1i)

        wa = (x1 - px) * (y1 - py)
        wb = (x1 - px) * (py - y0)
        wc = (px - x0) * (y1 - py)
        wd = (px - x0) * (py - y0)

        wa = tf.cast(wa, tf.float32)
        wb = tf.cast(wb, tf.float32)
        wc = tf.cast(wc, tf.float32)
        wd = tf.cast(wd, tf.float32)

        out = Ia * wa + Ib * wb + Ic * wc + Id * wd
        return out[..., None]

    return sample(source, px, py)

# ==============================
# 6. Test the model
# ==============================
if __name__ == "__main__":
    print("Building improved PICoLT model...")
    model = build_improved_picolt()
    model.summary()
    
    # Test forward pass
    dummy_input = tf.random.normal((2, 128, 128, 1))
    params_pred, source_pred = model(dummy_input)
    
    print(f"\nOutput shapes:")
    print(f"Parameters: {params_pred.shape}")
    print(f"Source: {source_pred.shape}")
    
    # Test perceptual loss
    print("\nTesting perceptual loss...")
    perceptual_loss = PerceptualLoss()
    loss = perceptual_loss(source_pred, source_pred)
    print(f"Perceptual loss (should be ~0): {loss.numpy()}")
    
    print("\nModel built successfully!")