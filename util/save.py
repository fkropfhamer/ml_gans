import tensorflow as tf
import os

def save_model(model, name):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, f'../models/{name}.h5')
    lite_filename = os.path.join(dirname, f'../tflite_models/{name}.tflite')

    model.save(filename)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(lite_filename, "wb").write(tflite_model)
