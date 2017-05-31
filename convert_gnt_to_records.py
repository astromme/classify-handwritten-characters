#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

import utils.gnt
import utils.tagcode

hyper_params = {
    'image_width' : 28,
    'image_height' : 28,
    'max_output_classes' : 4000,
    'max_imput_samples' : 50000
}

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def build_tagcode_index(samples, tagcode_index={}, label_keys=[]):
    index = len(label_keys)
    for strokes, tagcode in samples:
        if tagcode in tagcode_index:
            pass
        else:
            key = utils.tagcode.tagcode_to_unicode(tagcode)
            tagcode_index[tagcode] = index
            label_keys.append(key)
            index += 1

    return tagcode_index, label_keys



def load(dir_path, tagcode_index={}, label_keys=[]):
    tagcodes = {}
    samples = []
    num_processed = 0
    num_selected = 0
    print('loading samples')

    for bitmap, tagcode in utils.gnt.read_gnt_in_directory(dir_path):
        num_processed += 1
        if num_processed % 1000 == 0:
            print("processed {} samples, selected {} samples from {} classes".format(num_processed, num_selected, len(tagcodes)))

        if len(tagcodes) < hyper_params['max_output_classes'] or tagcode in tagcodes:
            samples.append((bitmap, tagcode))
            tagcodes[tagcode] = True
            num_selected += 1

            if num_selected >= hyper_params['max_imput_samples']:
                break

    print("selected {} samples".format(num_selected))
    tagcode_index, label_keys = build_tagcode_index(samples, tagcode_index, label_keys)

    samples = [(bitmap, tagcode_index[tagcode]) for bitmap, tagcode in samples]
    return samples, tagcode_index, label_keys


def write_tf_records():
    from tqdm import tqdm

    samples, tagcode_index, label_keys = load('HWDB1.1tst_gnt')

    # one MUST randomly shuffle data before putting it into one of these
    # formats. Without this, one cannot make use of tensorflow's great
    # out of core shuffling.
    np.random.shuffle(samples)

    with open("label_keys.list", 'wb') as f:
        for key in label_keys:
            f.write((key + '\n').encode('utf-8'))

    writer = tf.python_io.TFRecordWriter("hwdb1.1.tfrecords")

    def write_example(writer, bitmap, label):
        # construct the Example proto boject
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(label)),
            'height': _int64_feature(bitmap.shape[0]),
            'width': _int64_feature(bitmap.shape[1]),
            'depth': _int64_feature(1),
            'image_raw': _bytes_feature(bitmap.tostring()),
        }));

        serialized = example.SerializeToString()
        writer.write(serialized)

    # iterate over each example
    # wrap with tqdm for a progress bar
    for bitmap, label in tqdm(samples):
        write_example(writer, bitmap, label)



if __name__ == "__main__":
    write_tf_records()
