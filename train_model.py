import tensorflow as tf
from model.character_recognizer_model import CharacterRecognizerModel
from libgnt.gnt import samples_from_gnt, samples_from_directory
from libgnt.character_index import character_index

import numpy as np
import random

from tqdm import tqdm

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

    print(image.shape)
    image = tf.reshape(image, image_shape, name='image')
    label = tf.reshape(label, [1])

    image = 255 - image
    image = tf.cast(image, tf.float32)
    image = image / 255

    image = tf.image.resize_with_pad(
        image=image,
        target_height=28,
        target_width=28)

    return image, label

def apply_transforms(image, label):

    #TODO: Fix the transforms
    #pad the image by up to 3
    # padding_x = tf.random.uniform([1], 1, 3, name='padding_x')
    # padding_y = tf.random.uniform([1], 1, 3, name='padding_y')
    # image = tf.image.resize_with_crop_or_pad(
    #     image=image,
    #     target_height=tf.cast(tf.cast(height, tf.float32)*padding_y[0], tf.int32),
    #     target_width=tf.cast(tf.cast(width, tf.float32)*padding_x[0], tf.int32),
    # )

    # rotate by up to 60 degrees in either direction
    # angles = tf.random_uniform([1], -1/3, 1/3, name='random_rotation_radians')
    # image = tf.contrib.image.rotate(image, angles)

    # image = tf.image.resize_with_crop_or_pad(
    #     image=resized_image,
    #     target_height=28,
    #     target_width=28)

    return image, label

train_ds = tf.data.TFRecordDataset(filenames = ['hwdb.train.tfrecords'])
train_ds = train_ds.map(parse_record)
train_ds = train_ds.map(apply_transforms)
train_ds = train_ds.batch(32)

test_ds = tf.data.TFRecordDataset(filenames = ['hwdb.test.tfrecords'])
test_ds = test_ds.map(parse_record)
test_ds = test_ds.map(apply_transforms)
test_ds = test_ds.batch(32)


hp = {
    'n_classes': len(character_index),
    # 'n_classes': 10,
    'keep_prob': 1.0,
}

# Create an instance of the model
model = CharacterRecognizerModel(**hp)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])


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
    print(predictions.shape)
    print(predictions[0])
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


EPOCHS = 50

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in tqdm(train_ds):
    train_step(images, labels)

  for test_images, test_labels in tqdm(test_ds):
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )

  model.save('trained_model.tf', save_format='tf')
