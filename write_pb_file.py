#!/usr/bin/env python3

from gnt_model import model, error_rate, IMAGE_HEIGHT, IMAGE_WIDTH, PIXEL_DEPTH

import os
import sys
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

def main():
    if len(sys.argv) != 3:
        print('Usage: {} checkpoint_path output_dir'.format(sys.argv[0]))
        sys.exit()

    _, checkpoint_path, output_dir = sys.argv

    node_image_raw = tf.placeholder("float", shape=[None, 784], name="input")

    node_normalized_image = tf.reshape(node_image_raw, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 1]) / PIXEL_DEPTH - 0.5

    node_logits = model(node_normalized_image)
    node_predictions = tf.nn.softmax(node_logits, name="output")

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        print('loading model')
        sess.run(init_op)
        saver.restore(sess, checkpoint_path)

        pb_filename = os.path.join(output_dir, 'frozen_character_model_graph.pb')
        graph_def = tf.get_default_graph().as_graph_def()

        for node in graph_def.node:
            node.device = ""

        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            graph_def,
            ['output'])

        print('writing {}'.format(pb_filename))

        with gfile.GFile(pb_filename, "wb") as f:
            f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    main()
