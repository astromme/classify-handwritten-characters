#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

from libgnt.gnt import samples_from_directory
from libgnt.character_index import character_index

import random

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

from tqdm import tqdm

def load(dir_paths):
    samples = []
    for dir_path in dir_paths:
        for bitmap, character in tqdm(samples_from_directory(dir_path)):
            samples.append((bitmap, character_index.index(character)))

    return samples

def write_tf_records():
    from tqdm import tqdm
    import sys

    if len(sys.argv) < 3:
        print(f'usage: {sys.argv[0]} output_tfrecords_filename gnt_folder [gnt_folder2 ...]')
        print(f'e.g. $ {sys.argv[0]} characters.train.tfrecord Gnt1.0TrainPart1')
        sys.exit()

    output_tfrecords_filename = sys.argv[1]
    gnt_folders = sys.argv[2:]

    samples = load(gnt_folders)

    # one MUST randomly shuffle data before putting it into one of these
    # formats. Without this, one cannot make use of tensorflow's great
    # out of core shuffling.
    np.random.shuffle(samples)

    writer = tf.io.TFRecordWriter(output_tfrecords_filename)

    def write_example(writer, bitmap, label):
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(label)),
            'height': _int64_feature(bitmap.shape[0]),
            'width': _int64_feature(bitmap.shape[1]),
            'depth': _int64_feature(1),
            'image_raw': _bytes_feature(bitmap.tobytes()),
        }));

        serialized = example.SerializeToString()
        writer.write(serialized)

    # iterate over each example
    # wrap with tqdm for a progress bar
    for bitmap, label in tqdm(samples):
        write_example(writer, bitmap, label)

    print(f"{len(samples)} samples written")


if __name__ == "__main__":
    write_tf_records()
