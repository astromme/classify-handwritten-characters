import tensorflow as tf
from model.character_recognizer_model import CharacterRecognizerModel
from libgnt.gnt import samples_from_gnt, samples_from_directory
from libgnt.character_index import character_index, write_character_index
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import tensorflow.python.framework.errors_impl as errors_impl

import numpy as np
import random
import math
import sys

from tqdm import tqdm

import datetime
LOG_DIR = 'logs/gradient_tape/'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = LOG_DIR + current_time + '/train'
test_log_dir = LOG_DIR + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48

EPOCHS = 50

def parse_record(raw_example):
    example = tf.io.parse_example(
        raw_example,
        features={
          'label': tf.io.FixedLenFeature([], tf.int64),
          'height': tf.io.FixedLenFeature([], tf.int64),
          'width': tf.io.FixedLenFeature([], tf.int64),
          'depth': tf.io.FixedLenFeature([], tf.int64),
          'image_raw': tf.io.FixedLenFeature([], tf.string),
          }
    )

    # Convert from a scalar string tensor (whose single string has
    # length IMAGE_PIXELS) to a uint8 tensor with shape
    # [IMAGE_PIXELS].
    image = tf.io.decode_raw(example['image_raw'], tf.uint8)
    # image = tf.image.convert_image_dtype(image, tf.float32)

    label = tf.cast(example['label'], tf.int32)
    height = tf.cast(example['height'], tf.int32)
    width = tf.cast(example['width'], tf.int32)
    depth = tf.cast(example['depth'], tf.int32)

    image_shape = tf.stack([height, width, depth])

    image = tf.reshape(image, image_shape, name='image')
    label = tf.reshape(label, [1])

    image = 255 - image
    image = tf.cast(image, tf.float32)
    image = image / 255

    image = tf.image.resize_with_pad(
        image=image,
        target_height=IMAGE_HEIGHT,
        target_width=IMAGE_WIDTH)

    return image, label

g = tf.random.Generator.from_seed(1)

def apply_transforms(image, label):
    # random rotation
    max_rotation_radians = 0.1 * 3.14
    random_rotation = g.uniform([], minval=-max_rotation_radians, maxval=max_rotation_radians, dtype=tf.dtypes.float32)
    image = tfa.image.rotate(image, random_rotation)

    # random padding up to 25% of the image size
    max_padding_pixels = math.floor(0.25 * IMAGE_WIDTH)
    padding_x = g.uniform([], minval=0, maxval=max_padding_pixels, dtype=tf.dtypes.int32)
    padding_y = g.uniform([], minval=0, maxval=max_padding_pixels, dtype=tf.dtypes.int32)

    image = tf.image.resize_with_crop_or_pad(
        image,
        target_height=IMAGE_HEIGHT + padding_y,
        target_width=IMAGE_WIDTH + padding_x
    )

    image = tf.image.resize_with_pad(image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)

    # random translation
    epsilon = 1
    translation_x = g.uniform([], minval=-padding_x-epsilon, maxval=padding_x, dtype=tf.dtypes.int32)
    translation_y = g.uniform([], minval=-padding_y-epsilon, maxval=padding_y, dtype=tf.dtypes.int32)
    image = tfa.image.translate(image, [translation_x, translation_y])

    # random squishing / stretching
    max_squish_factor = 0.25
    new_width = tf.math.floor(IMAGE_WIDTH * g.uniform([], minval=1-max_squish_factor, maxval=1, dtype=tf.dtypes.float32))
    new_height = tf.math.floor(IMAGE_HEIGHT * g.uniform([], minval=1-max_squish_factor, maxval=1, dtype=tf.dtypes.float32))
    image = tf.image.resize(image, [new_height, new_width])
    image = tf.image.resize_with_pad(image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)

    image = tf.image.random_contrast(image, 0.3, 2.0)

    return image, label

wrapped_apply_transforms = lambda image, label: tf.py_function(func=apply_transforms, inp=[image, label], Tout=[tf.float32, tf.int64])


# train_ds = tf.data.TFRecordDataset(filenames = ['hwdb.train.tfrecords'])
# train_ds = train_ds.map(parse_record)
# train_ds = train_ds.map(wrapped_apply_transforms)
# train_ds = train_ds.batch(32)
#
# test_ds = tf.data.TFRecordDataset(filenames = ['hwdb.test.tfrecords'])
# test_ds = test_ds.map(parse_record)
# test_ds = test_ds.map(wrapped_apply_transforms)
# test_ds = test_ds.batch(32)

print("loading kaggle hwdb")
builder = tfds.ImageFolder('trainingdata/kaggle-hwdb-images-archive/')
# builder = tfds.ImageFolder('trainingdata/generated-from-fonts/')
# builder = tfds.ImageFolder('trainingdata/kaggle-mini/')

write_character_index(builder.info.features['label'].names)

def parse_tfds_record(example):
    image = example["image"]
    image = 255 - image
    image = tf.cast(image, tf.float32)
    image = image / 255

    image = tf.image.resize_with_pad(
        image=image,
        target_height=IMAGE_HEIGHT,
        target_width=IMAGE_WIDTH)

    return image, example["label"]


# print(builder.info)  # num examples, labels... are automatically calculated
train_ds = builder.as_dataset(split='Train', shuffle_files=False)
# sys.quit()
# it = iter(train_ds)
# example = next(it)
# print(f'example: {example}')
# print(train_ds)
# print(builder.info.features['label'].names)

train_ds = train_ds.map(parse_tfds_record)
train_ds = train_ds.map(wrapped_apply_transforms)


import matplotlib
matplotlib.rcParams['font.family'] = ['Heiti TC']
# tfds.show_examples(train_ds, builder.info)
train_ds = train_ds.batch(128)

test_ds = builder.as_dataset(split='Test', shuffle_files=False)
test_ds = test_ds.map(parse_tfds_record)
# test_ds = test_ds.map(wrapped_apply_transforms)
# tfds.show_examples(test_ds, builder.info)
test_ds = test_ds.batch(128)



# sizes = []
#
# for example in tqdm(train_ds):
#     sizes.append(f'{example}')

# print(sizes)

# sys.exit()
# it = iter(train_ds)
# images, labels = next(it)
# from utils.show_tf_image import show_tf_image
# show_tf_image(images[0])
# sys.exit()

hp = {
    'n_classes': len(character_index),
    'keep_prob': 0.7,
}

# Create an instance of the model
if sys.argv[1] == '--resume':
    print(f'resuming training of model')
    model = tf.keras.models.load_model('trained_model.tf')
else:
    model = CharacterRecognizerModel(**hp)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])

# tb_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)
# tb_callback.set_model(model) # Writes the graph to tensorboard summaries using an internal file writer
tf.summary.trace_on(graph=True, profiler=True)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)

  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)



for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()


    for i in range(1):
        print(i)
        tqdm_train_ds = tqdm(train_ds)
        it = iter(tqdm_train_ds)
        while True:
            try:
                images, labels = next(it)
                train_step(images, labels)
            except errors_impl.InvalidArgumentError as e:
                print(f'ignoring {e} and continuing with next batch...')
                continue
            except StopIteration:
                break
    with train_summary_writer.as_default():
        # if epoch == 0:
        #     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=LOG_DIR)
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    try:
        for test_images, test_labels in tqdm(test_ds):
            test_step(test_images, test_labels)
    except errors_impl.InvalidArgumentError as e:
        print(f'ignoring {e} and continuing with next epoch...')

    with test_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )

    model.save('trained_model.tf', save_format='tf')
