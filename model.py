# model.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config

# --- Custom Layers and Helper Functions ---

class Patches(layers.Layer):

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

def patch_to_image(patches, patch_size, img_size, channels):
    
    batch = tf.shape(patches)[0]
    h = img_size // patch_size
    patches = tf.reshape(patches, [batch, h, h, patch_size, patch_size, channels])
    patches = tf.transpose(patches, [0, 1, 3, 2, 4, 5])
    return tf.reshape(patches, [batch, img_size, img_size, channels])

def transformer_block(x, num_heads, ff_dim, dropout_rate):
    
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=x.shape[-1]
    )(x, x)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    ffn_output = layers.Dense(ff_dim, activation='relu')(out1)
    ffn_output = layers.Dense(x.shape[-1])(ffn_output)
    ffn_output = layers.Dropout(dropout_rate)(ffn_output)
    
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


# --- Model Building Function ---

def build_autoencoder():

    inputs = keras.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.CHANNELS))
    
    # Encoder
    patches = Patches(config.PATCH_SIZE)(inputs)
    num_patches = (config.IMAGE_SIZE // config.PATCH_SIZE) ** 2
    encoded_tokens = PatchEncoder(num_patches, config.PROJECTION_DIM)(patches)

    for _ in range(config.NUM_TRANSFORMER_LAYERS):
        encoded_tokens = transformer_block(
            encoded_tokens, config.NUM_HEADS, config.FF_DIM, 0.1
        )

    latent = encoded_tokens

    # Decoder
    decoded_tokens = latent
    for _ in range(config.NUM_TRANSFORMER_LAYERS):
        decoded_tokens = transformer_block(
            decoded_tokens, config.NUM_HEADS, config.FF_DIM, 0.1
        )
        
    patch_dim = config.PATCH_SIZE * config.PATCH_SIZE * config.CHANNELS
    reconstructed_patches = layers.Dense(patch_dim)(decoded_tokens)
    
    outputs = layers.Lambda(lambda x: patch_to_image(
        x, config.PATCH_SIZE, config.IMAGE_SIZE, config.CHANNELS
    ))(reconstructed_patches)

    return keras.Model(inputs=inputs, outputs=outputs)