import tensorflow as tf
from tensorflow import keras

def load_and_prepare_data():

    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    return x_train, x_test