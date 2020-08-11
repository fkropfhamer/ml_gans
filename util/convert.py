import os
import tensorflow as tf
from tensorflow import keras
import glob

dirname = os.path.dirname(__file__)

def convert_keras_model_to_tflite(filename):
    model_filename = os.path.join(dirname, f'../models/{filename}')
    lite_filename = os.path.join(dirname, f'../tflite_models/{filename}.tflite')

    model = keras.models.load_model(model_filename)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(lite_filename, "wb").write(tflite_model)


if __name__ == "__main__":
    models = os.listdir(os.path.join(dirname, '../models'))

    for model in models:
        if model.endswith(".h5"):
            convert_keras_model_to_tflite(model)
