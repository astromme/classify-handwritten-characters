#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

from utils import read_pot_in_directory, tagcode_to_unicode


hyper_params = {
    'max_seq_length' : 200,
    'input_samples' : 1000000,
    'max_output_classes' : 10000,
}

def create_example(strokes, tagcode, tagcode_index):
    inputs = []

    for stroke in strokes:
        for point in stroke:
            inputs.append(list(point) + [0]) # [x, y, pen_state]
        inputs.append([0, 0, 1])

    while len(inputs) < hyper_params['max_seq_length']:
        inputs.append([0, 0, 1])

    if len(inputs) > hyper_params['max_seq_length']:
        print('truncating sequence of length {}'.format(len(inputs)))
        inputs = inputs[:hyper_params['max_seq_length']]

    return np.asarray(inputs, dtype=np.float32).flatten(), tagcode_index[tagcode]

def build_tagcode_index(samples, tagcode_index={}, label_keys=[]):
    index = len(label_keys)
    for strokes, tagcode in samples:
        if tagcode in tagcode_index:
            pass
        else:
            key = tagcode_to_unicode(tagcode)
            tagcode_index[tagcode] = index
            label_keys.append(key)
            index += 1

    return tagcode_index, label_keys



def load(dir_path, tagcode_index={}, label_keys=[]):
    tagcodes = {}
    data = []
    tags = []
    samples = []
    num_processed = 0
    num_selected = 0
    print('loading samples')

    for strokes, tagcode in read_pot_in_directory(dir_path):
        num_processed += 1
        if num_processed % 1000 == 0:
            print("processed {} samples, selected {} samples from {} classes".format(num_processed, num_selected, len(tagcodes)))
        if len(tagcodes) < hyper_params['max_output_classes'] or tagcode in tagcodes:
            samples.append((strokes, tagcode))
            tagcodes[tagcode] = True
            num_selected += 1

            if num_selected >= hyper_params['input_samples']:
                break

    np.random.shuffle(samples)

    print("selected {} samples".format(num_selected))
    tagcode_index, label_keys = build_tagcode_index(samples, tagcode_index, label_keys)
    #print(tagcode_index)


    for strokes, tagcode in samples:
        example, tag = create_example(strokes, tagcode, tagcode_index)
        data.append(example)
        tags.append(tag)

    data = np.array(data)
    tags = np.array(tags)


    return data, tags, tagcode_index, label_keys


def write_tf_records():
    from tqdm import tqdm

    training_data, training_labels, tagcode_index, label_keys = load('OLHWDB1.1trn_pot')
    testing_data, testing_labels, tagcode_index, label_keys = load('OLHWDB1.1tst_pot', tagcode_index, label_keys)


    training = list(zip(training_data, training_labels))
    testing = list(zip(testing_data, testing_labels))

    # one MUST randomly shuffle data before putting it into one of these
    # formats. Without this, one cannot make use of tensorflow's great
    # out of core shuffling.
    np.random.shuffle(training)
    np.random.shuffle(testing)

    with open("label_keys.list", 'wb') as f:
        for key in label_keys:
            f.write((key + '\n').encode('utf-8'))

    writer = tf.python_io.TFRecordWriter("hwdb1.1.tfrecords")

    def write_example(writer, points, label):
        # construct the Example proto boject
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
              # Features contains a map of string to Feature proto objects
              feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label])),
                'points': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=points.astype("int64"))),
        }))
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)

    # iterate over each example
    # wrap with tqdm for a progress bar
    for points, label in tqdm(training):
        write_example(writer, points, label)

    for points, label in tqdm(testing):
        write_example(writer, points, label)



if __name__ == "__main__":
    write_tf_records()
