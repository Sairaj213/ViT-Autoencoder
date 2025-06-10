# train.py

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import config
import data_utils
import model

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(config.HISTORY_PLOT_PATH)
    plt.close()
    print(f"Training history plot saved to {config.HISTORY_PLOT_PATH}")

def train_model():

    # 1. Load Data
    print("Loading and preparing data...")
    x_train, x_test = data_utils.load_and_prepare_data()

    # 2. Build Model
    print("Building the autoencoder model...")
    autoencoder = model.build_autoencoder()

    # 3. Compile Model
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    autoencoder.compile(optimizer=optimizer, loss="mse")
    autoencoder.summary()
    
    # 4. Train Model
    print("Starting model training...")
    history = autoencoder.fit(
        x_train, x_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(x_test, x_test)
    )

    # 5. Save Model and History
    print("\nTraining complete.")
    autoencoder.save(config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")
    
    plot_training_history(history)

    return autoencoder, history, x_test