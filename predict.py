import tensorflow as tf
import numpy as np
import sys
from libgnt.character_index import character_index
from utils.show_tf_image import show_tf_image
from utils.array_top_n_indexes import array_top_n_indexes
import os

def main():
    if len(sys.argv) < 3:
        print("usage: predict.py MODEL PNG_FILE")
        sys.exit()

    _, model_filename, png_filename = sys.argv

    model = tf.keras.models.load_model(model_filename)
    image = tf.keras.utils.load_img(png_filename, color_mode='grayscale')
    input_arr = tf.keras.utils.img_to_array(image)

    input_arr = 255 - input_arr
    input_arr = tf.cast(input_arr, tf.float32)
    input_arr = input_arr / 255
    input_arr = tf.image.resize_with_pad(input_arr, 28, 28)

    predictions = model(np.array([input_arr]), training=False)
    for char_index in array_top_n_indexes(predictions[0], 5):
        print(f'c: {character_index[char_index]}, v: {predictions[0][char_index]}')
    # predicted_char_num = np.argmax(predictions, axis = 1)
    top_5_predictions = [character_index[c] for c in array_top_n_indexes(predictions[0], 5)]
    print(top_5_predictions)

    show_tf_image(input_arr, f'"{os.path.basename(png_filename)}" predictions: {top_5_predictions}')


if __name__ == "__main__":
    main()
