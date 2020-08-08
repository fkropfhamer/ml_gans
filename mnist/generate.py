import tensorflow as tf
from tensorflow import keras
import numpy as np


def main():
    generator = keras.models.load_model('./mnist_model.h5')

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    generated_image = np.array(generated_image[0, :, :, 0])

    generated_image = generated_image * 127.5 + 127.5

    generated_image = generated_image.astype(np.uint8)

    print(generated_image)
    print(generated_image.shape)

    
    


if __name__ == "__main__":
    main()
