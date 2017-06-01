#!/usr/bin/env python3

from gnt_model import model, IMAGE_HEIGHT, IMAGE_WIDTH, PIXEL_DEPTH

import sys
import numpy
import png
import skimage.io as io
import tensorflow as tf

_, model_path, image_filename = sys.argv

with open(image_filename, 'rb') as f:
    direct = png.Reader(f).asDirect()
    print(direct)

    mapped = map(numpy.uint8, direct[2])
    print(mapped)

    image_2d = numpy.vstack(list(mapped))


image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=tf.int32)
x = tf.placeholder(tf.float32, shape=(1, None, None, 1))
resized_x = tf.image.resize_images(images=x,
                                          size=image_size_const
                                          )
resized_padded_x = tf.image.resize_image_with_crop_or_pad(image=resized_x,
                                           target_height=IMAGE_HEIGHT,
                                           target_width=IMAGE_WIDTH)

scaled_x = resized_padded_x / PIXEL_DEPTH - 0.5


logits = model(scaled_x)
prediction = tf.nn.softmax(logits)

saver = tf.train.Saver()

with open('label_keys.list', encoding='utf8') as f:
    labels = f.readlines()

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, model_path)
    print("Model restored.")

    height, width = image_2d.shape
    predictions = sess.run([prediction], feed_dict={x: image_2d.reshape((1, height, width, 1))})

    for index in numpy.argsort(predictions[0][0])[-11:]:
        print(labels[index])


#images_batch / PIXEL_DEPTH - 0.5
