# evaluation.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import config

def display_reconstructions(model, data, n=10):

    reconstructed = model.predict(data[:n])
    
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(data[i])
        plt.title("Original")
        plt.axis("off")
        
        # Reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(np.clip(reconstructed[i], 0, 1))
        plt.title("Recon")
        plt.axis("off")
        
    plt.suptitle("Original vs. Reconstructed Images", fontsize=16)
    plt.savefig(config.RECONSTRUCTION_PLOT_PATH)
    plt.close()
    print(f"Reconstruction plot saved to {config.RECONSTRUCTION_PLOT_PATH}")

def evaluate_and_reconstruct_image(image_input, model):

    width, height = (config.IMAGE_SIZE, config.IMAGE_SIZE)

    # 1. Load and resize
    if isinstance(image_input, str):
        img = load_img(image_input, target_size=(height, width))
        img = img_to_array(img) / 255.0
    else:
        raise ValueError("Unsupported image input type. Please provide a file path.")

    img_original = np.expand_dims(img, axis=0)  # (1, H, W, 3)

    # 2. Reconstruct
    print(f"\nReconstructing image from: {config.SAMPLE_IMAGE_PATH}")
    reconstructed = model.predict(img_original)

    # 3. Calculate Metrics
    original = np.squeeze(img_original)
    reconstructed = np.squeeze(reconstructed)
    mse = np.mean((original - reconstructed) ** 2)
    psnr_val = psnr(original, reconstructed, data_range=1.0)
    ssim_val = ssim(original, reconstructed, data_range=1.0, channel_axis=2)

    # 4. Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(original, 0, 1))
    plt.title("Original Sample")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(reconstructed, 0, 1))
    plt.title("Reconstructed Sample")
    plt.axis("off")
    plt.show()

    # 5. Print report
    print("\n--- Image Quality Metrics ---")
    print(f" - MSE:  {mse:.6f}")
    print(f" - PSNR: {psnr_val:.2f} dB")
    print(f" - SSIM: {ssim_val:.4f}")