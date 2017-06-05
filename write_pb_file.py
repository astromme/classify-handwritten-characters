#!/usr/bin/env python3

from gnt_model import model, error_rate, IMAGE_HEIGHT, IMAGE_WIDTH, PIXEL_DEPTH

import sys
import tensorflow as tf

def main():
    if len(sys.argv) != 3:
        print('Usage: {} modelpath outputdir'.format(sys.argv[0]))
        sys.exit()

    _, model_path, output_dir = sys.argv

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
        saver.restore(sess, model_path)

        pb_filename = 'character_model_graph.pb.txt'
        print('writing {}'.format(pb_filename))
        graph_def = tf.get_default_graph().as_graph_def()
        tf.train.write_graph(graph_def, output_dir, pb_filename, as_text=True)

if __name__ == '__main__':
    main()
