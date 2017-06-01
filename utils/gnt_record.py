#!/usr/bin/env python3

import tensorflow as tf

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_DEPTH = 1
BATCH_SIZE = 1
NUM_EPOCHS = 1
NUM_SAMPLES = 4222

tfrecords_filename = "hwdb1.1.tfrecords"

def read_and_decode(filename_queue):

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)

    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    depth = tf.cast(features['depth'], tf.int32)

    image_shape = tf.stack([height, width, depth])

    image = tf.reshape(image, image_shape)
    label = tf.reshape(label, [1])

    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=tf.int32)

    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.

    longer_side = tf.maximum(height, width)

    resized_image = tf.image.resize_images(images=image,
                                          size=image_size_const
                                          )
    resized_padded_image = tf.image.resize_image_with_crop_or_pad(image=resized_image,
                                           target_height=IMAGE_HEIGHT,
                                           target_width=IMAGE_WIDTH)


    images, labels = tf.train.shuffle_batch( [resized_padded_image, label],
                                                 batch_size=BATCH_SIZE,
                                                 capacity=30,
                                                 num_threads=2,
                                                 shapes=[(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH), [1]],
                                                 min_after_dequeue=10)

    return images, tf.reshape(labels, [BATCH_SIZE])


def main():
    import skimage.io as io
    with open('label_keys.list') as f:
        labels = f.readlines()

    filename_queue = tf.train.string_input_producer(
        [tfrecords_filename], num_epochs=NUM_EPOCHS)

    # Even when reading in multiple threads, share the filename
    # queue.
    image_queue, label_queue = read_and_decode(filename_queue)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session()  as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_steps = NUM_EPOCHS*(NUM_SAMPLES//BATCH_SIZE)
        for step in range(num_steps):
            step += 1
            img, label = sess.run([image_queue, label_queue])
            print("step {} of {}".format(step, num_steps))

            # print('current batch')
            #
            # # We selected the batch size of two
            # # So we should get two image pairs in each batch
            # # Let's make sure it is random
            #
            # print(labels[label[0]])
            #
            # io.imshow(img[0, :, :, :].reshape([128, 128]))
            # io.show()
            #
            #
            # print(labels[label[1]])
            #
            # io.imshow(img[1, :, :, :].reshape([128, 128]))
            # io.show()

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
