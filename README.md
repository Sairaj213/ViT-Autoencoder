# Transformer Autoencoder for Image Reconstruction

This project implements a Transformer-based Autoencoder using TensorFlow and Keras to reconstruct images from the CIFAR-10 dataset. The architecture uses a Vision Transformer (ViT) style of patching images, encoding them with Transformer blocks, and then decoding them back into images.

## Project Structure

```
.
├── run.py                  # Main entry point to run the project
├── config.py               # All hyperparameters and file paths
├── data_utils.py           # Data loading and preprocessing for CIFAR-10
├── model.py                # Contains the Keras model architecture and custom layers
├── train.py                # Handles the model training loop and saving logic
├── evaluation.py           # Functions for visualizing results and calculating metrics
├── requirements.txt        # Project dependencies
├── sample_image.jpg        # A sample image for evaluation
└── README.md               # This file
```

## Setup and Installation

Follow these steps to set up the project environment.

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd transformer_autoencoder
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    If you have a GPU, ensure you have the correct version of CUDA and cuDNN installed and use `tensorflow[and-cuda]` or `tensorflow-gpu` if needed.

## How to Run

To run the entire pipeline—including data loading, training, model saving, and evaluation—simply execute the `run.py` script:

```bash
python run.py
```

The script will:
1.  Load and prepare the CIFAR-10 dataset.
2.  Build and compile the Transformer Autoencoder model.
3.  Train the model for the number of epochs specified in `config.py`.
4.  Save the trained model to `transformer_autoencoder.keras`.
5.  Generate and save a plot of the training history (`training_history.png`).
6.  Load the saved model and generate reconstructions of test set images (`test_reconstructions.png`).
7.  Evaluate a single sample image (`sample_image.jpg`) and print its quality metrics (MSE, PSNR, SSIM).

## Configuration

All key parameters can be modified in `config.py`. This includes:
- **Model architecture:** `IMAGE_SIZE`, `PATCH_SIZE`, `PROJECTION_DIM`, etc.
- **Training settings:** `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`.
- **File paths:** Location for saving the model and output plots.

To test with your own image, replace `sample_image.jpg` with your image and update the `SAMPLE_IMAGE_PATH` in `config.py`.

## Outputs

After running `run.py`, the following files will be generated in your project directory:
- `transformer_autoencoder.keras`: The saved, trained Keras model.
- `training_history.png`: A plot of the training and validation loss over epochs.
- `test_reconstructions.png`: A comparison of original and reconstructed images from the test set.

![Test Reconstructions](test_reconstructions.png)
![Training History](training_history.png)