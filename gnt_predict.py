#!/usr/bin/env python3

from gnt_model import model, error_rate, IMAGE_HEIGHT, IMAGE_WIDTH, PIXEL_DEPTH, NUM_EPOCHS
from utils.gnt_record import read_and_decode

import sys
import numpy
import png
import skimage.io as io
import tensorflow as tf

_, model_path, image_filename = sys.argv

with open(image_filename, 'rb') as f:
    direct = png.Reader(f).asDirect()
    mapped = map(numpy.uint8, direct[2])
    image_2d = numpy.vstack(list(mapped))


image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=tf.int32)
x = tf.placeholder(tf.uint8, shape=(1, None, None, 1))
resized_x = tf.image.resize_images(images=x,
                                          size=image_size_const
                                          )
resized_padded_x = tf.image.resize_image_with_crop_or_pad(image=resized_x,
                                           target_height=IMAGE_HEIGHT,
                                           target_width=IMAGE_WIDTH)

scaled_x = resized_padded_x / PIXEL_DEPTH - 0.5

logits = model(scaled_x)
predict_x = tf.nn.softmax(logits)

saver = tf.train.Saver()

with open('label_keys.list', encoding='utf8') as f:
    labels_to_char = f.read().split('\n')

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    # Restore variables from disk.

    sess.run(init_op)
    saver.restore(sess, model_path)
    print("Model restored.")

    height, width = image_2d.shape
    predictions, images_2d_resized = sess.run([predict_x, resized_padded_x], feed_dict={x: image_2d.reshape((1, height, width, 1))})
    #
    # io.imshow(image_2d.reshape(height, width))
    # io.show()

    image_2d_resized = images_2d_resized[0, :, :, :]
    h, w, _ = image_2d_resized.shape

    for prediction in predictions:
        top5 = reversed(numpy.argsort(prediction)[-10:])
        print([labels_to_char[i] for i in top5])

        index = numpy.argmax(prediction)
        print(index, labels_to_char[index])
