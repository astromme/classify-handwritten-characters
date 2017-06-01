#!/usr/bin/env python3

from gnt_model import model, error_rate, IMAGE_HEIGHT, IMAGE_WIDTH, PIXEL_DEPTH, NUM_EPOCHS
from utils.gnt_record import read_and_decode, BATCH_SIZE

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
x = tf.placeholder(tf.uint8, shape=(BATCH_SIZE, None, None, 1))
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

tfrecords_filename = "hwdb1.1.test.tfrecords.full"

filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=NUM_EPOCHS)


image_queue, label_queue = read_and_decode(filename_queue)
image_queue_normalized = image_queue / PIXEL_DEPTH - 0.5
queue_logits = model(image_queue_normalized)
queue_prediction = tf.nn.softmax(queue_logits)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
  labels=label_queue, logits=queue_logits))

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

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    images, l, normalized_images, labels, predictions = sess.run([image_queue, loss, image_queue_normalized, label_queue, queue_prediction],
        feed_dict={x: numpy.zeros((BATCH_SIZE, 1, 1, 1))})

    image1 = images[0, :, :, :]
    h, w, d = image1.shape

    import png
    #print(image1.reshape(h, w).astype(numpy.uint8))
    png.from_array(image1.reshape(h, w).astype(numpy.uint8), 'L').save("test.png")

    # io.imshow(image1.reshape(h, w))
    # io.show()
    #
    # io.imshow(normalized_images[0, :, :, :].reshape(28, 28))
    # io.show()

    for label, prediction in zip(labels, predictions):
        top5 = reversed(numpy.argsort(prediction)[-10:])
        print([labels_to_char[i] for i in top5])

        index = numpy.argmax(prediction)
        print(label, labels_to_char[label], index, labels_to_char[index], label == index)#, prediction)

    print(error_rate(predictions, labels))
    print(l)

    #image_2d = image_2d * 255
    print(image_2d.shape)
    height, width = image_2d.shape
    predictions, images_2d_resized = sess.run([predict_x, resized_padded_x], feed_dict={x: image_2d.reshape((1, height, width, 1))})
    #
    io.imshow(image_2d.reshape(height, width))
    io.show()

    image_2d_resized = images_2d_resized[0, :, :, :]
    print(image_2d_resized.shape)
    h, w, _ = image_2d_resized.shape
    png.from_array(image_2d_resized.reshape(h, w).astype(numpy.uint8), 'L').save("test.png")

    for prediction in predictions:
        print(prediction.shape)
        top5 = reversed(numpy.argsort(prediction[0])[-10:])
        print([labels_to_char[i] for i in top5])

        index = numpy.argmax(prediction)
        print(index, labels_to_char[index])

    # for index in numpy.argsort(predictions[0][0])[-1:]:
    #     print(labels_to_char[index])

    coord.request_stop()
    coord.join(threads)
#images_batch / PIXEL_DEPTH - 0.5
