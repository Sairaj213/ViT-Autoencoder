# --- Model Configuration ---
IMAGE_SIZE = 32
PATCH_SIZE = 4
PROJECTION_DIM = 64
NUM_TRANSFORMER_LAYERS = 4
NUM_HEADS = 4
FF_DIM = 128
CHANNELS = 3

# --- Training Configuration ---
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# --- File Paths ---
MODEL_PATH = "transformer_autoencoder.keras"
HISTORY_PLOT_PATH = "training_history.png"
RECONSTRUCTION_PLOT_PATH = "test_reconstructions.png"
SAMPLE_IMAGE_PATH = "sample_image.jpg" # Place your test image here