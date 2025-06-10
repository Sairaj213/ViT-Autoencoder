# run.py

import os
import tensorflow as tf
from tensorflow import keras

import config
import train
import evaluation
import model as model_def # Use an alias to avoid conflict with loaded model variable

def main():

    _, _, x_test = train.train_model()

    print("\nLoading saved model for evaluation...")
    custom_objects = {
        "Patches": model_def.Patches,
        "PatchEncoder": model_def.PatchEncoder
    }
    loaded_model = keras.models.load_model(config.MODEL_PATH, custom_objects=custom_objects)
    print("Model loaded successfully.")

    
    print("\nGenerating reconstructions from the test set...")
    evaluation.display_reconstructions(loaded_model, x_test)


    if not os.path.exists(config.SAMPLE_IMAGE_PATH):
        print(f"'{config.SAMPLE_IMAGE_PATH}' not found. Creating one from the test set.")
        sample_img_data = (x_test[0] * 255).astype("uint8")
        tf.keras.utils.save_img(config.SAMPLE_IMAGE_PATH, sample_img_data)
        
    evaluation.evaluate_and_reconstruct_image(config.SAMPLE_IMAGE_PATH, loaded_model)

    print("\n--- Process complete! ---")

if __name__ == "__main__":
    main()