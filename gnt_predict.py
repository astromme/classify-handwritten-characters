#!/usr/bin/env python3

from gnt_model import model, error_rate, IMAGE_HEIGHT, IMAGE_WIDTH, PIXEL_DEPTH, NUM_EPOCHS
from utils.gnt_record import read_and_decode

import sys
import numpy
import png
import skimage.io as io
import tensorflow as tf



image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=tf.int32)

node_image_raw = tf.placeholder(tf.string, shape=(None))
node_image = tf.image.decode_png(node_image_raw, channels=1, dtype=tf.uint8, name="load_image")
node_first_contrast = tf.image.adjust_contrast(images=node_image, contrast_factor=20)

node_resized_image = tf.image.resize_images(images=node_first_contrast, size=image_size_const, method=tf.image.ResizeMethod.AREA)
node_high_contrast = tf.image.adjust_contrast(images=node_resized_image, contrast_factor=1.5)
node_padded_image = tf.image.resize_image_with_crop_or_pad(image=node_high_contrast,
                                           target_height=IMAGE_HEIGHT,
                                           target_width=IMAGE_WIDTH)

node_normalized_image = tf.reshape(node_padded_image, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 1]) / PIXEL_DEPTH - 0.5

node_logits = model(node_normalized_image)
node_predictions = tf.nn.softmax(node_logits)

saver = tf.train.Saver()

with open('label_keys.list', encoding='utf8') as f:
    labels_to_char = f.read().split('\n')

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

_, model_path, image_filename = sys.argv

with open(image_filename, 'rb') as f:
    image_data = f.read()

with tf.Session() as sess:
    # Restore variables from disk.

    sess.run(init_op)
    saver.restore(sess, model_path)
    print("Model restored.")

    predictions, normalized_images = sess.run([node_predictions, node_high_contrast],
        feed_dict={node_image_raw: image_data})

    # image = normalized_images[0, :, :, :]
    # h, w, _ = image.shape
    #
    # io.imshow(image.reshape(h, w))
    # io.show()

    for prediction in predictions:
        top5 = reversed(numpy.argsort(prediction)[-10:])
        print([labels_to_char[i] for i in top5])

        index = numpy.argmax(prediction)
        print(index, labels_to_char[index])
