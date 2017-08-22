#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.contrib.layers import flatten
from utils.gnt_record import read_and_decode, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH
import time
import sys
import numpy
import tqdm

with open('label_keys.list', encoding='utf8') as f:
    labels = f.readlines()

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def StrommeNet3Layer(x, keep_prob, hp):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    initializer = tf.truncated_normal_initializer(mean=mu, stddev=sigma)

    # Layer 1: Convolutional.
    with tf.variable_scope('StrommeNet3Layer'):
        conv1_weights = tf.get_variable('conv1_weights', [5, 5, hp['num_channels'], hp['conv1_num_outputs']], initializer=initializer)
        variable_summaries(conv1_weights)

    x = tf.nn.conv2d(x, conv1_weights, strides=[1, 1, 1, 1], padding='VALID', name='conv1')
    x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # Activation.
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob=keep_prob, name='activation_dropout_1')

    # Layer 2: Convolutional.
    with tf.variable_scope('StrommeNet3Layer'):
        conv2_weights = tf.get_variable('conv2_weights', [5, 5, hp['conv1_num_outputs'], hp['conv2_num_outputs']], initializer=initializer)
        variable_summaries(conv2_weights)

    x = tf.nn.conv2d(x, conv2_weights, strides=[1,1,1,1], padding='VALID', name='conv2')
    x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # Activation.
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob=keep_prob, name='activation_dropout_2')

    layer2_flatten = flatten(x)

    # Layer 3: Convolutional.
    with tf.variable_scope('StrommeNet3Layer'):
        conv3_weights = tf.get_variable('conv3_weights', [3, 3, hp['conv2_num_outputs'], hp['conv3_num_outputs']], initializer=initializer)
        variable_summaries(conv3_weights)

    x = tf.nn.conv2d(x, conv3_weights, strides=[1,1,1,1], padding='VALID', name='conv3')
    x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


    # Activation.
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob=keep_prob, name='activation_dropout_3')

    # Flatten. Output = 512.
    flattened = tf.concat([flatten(x), layer2_flatten], 1)

    # Layer 3: Fully Connected.
    fully_connected1 = tf.layers.dense(inputs=flattened, units=384)

    # Activation.
    activation3 = tf.nn.relu(fully_connected1)

    # Layer 4: Fully Connected.
    fully_connected2 = tf.layers.dense(inputs=activation3, units=128)

    # Activation.
    activation4 = tf.nn.relu(fully_connected2)

    # Layer 5: Fully Connected.
    logits = tf.layers.dense(inputs=activation4, units=hp['n_classes'], name='logits')

    return logits


import math
import time
from datetime import datetime
import shutil
run_num = 0

def build_graph(hp):
    g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32, (None, hp['image_width'], hp['image_height'], hp['num_channels']), name='x')
        y = tf.placeholder(tf.int32, (None), name='y')
        keep_prob = tf.placeholder_with_default(1.0, [], name='keep_prob')
        one_hot_y = tf.one_hot(y, hp['n_classes'], name='one_hot_y')

        x_normalized = (x - hp['pixel_depth']//2) - 1
        logits = hp['model'](x_normalized, keep_prob, hp)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
        loss_operation = tf.reduce_mean(cross_entropy, name='loss_operation')
        tf.summary.scalar('loss', loss_operation)

        optimizer = tf.train.AdamOptimizer(learning_rate = hp['learning_rate'])
        training_operation = optimizer.minimize(loss_operation, name='training_operation')
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1), name='correct_prediction')
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_operation')
        tf.summary.scalar('accuracy', accuracy_operation)

        return g, x, y, keep_prob, logits, training_operation, accuracy_operation


def train(hp):
    graph, x, y, keep_prob, logits, training_operation, accuracy_operation = build_graph(hp)

    with graph.as_default():
        train_filename_queue = tf.train.string_input_producer(
            [hp['train_filename']], num_epochs=hp['epochs'])

        test_filename_queue = tf.train.string_input_producer(
            [hp['test_filename']], num_epochs=hp['epochs'])

        X_train_batch_op, y_train_batch_op = read_and_decode(train_filename_queue, hp['batch_size'])
        X_test_batch_op, y_test_batch_op = read_and_decode(test_filename_queue, hp['batch_size'])

    train_logs_dir = hp['logs_dir'] + '/{}-train'.format(hp['logs_prefix'])
    valid_logs_dir = hp['logs_dir'] + '/{}-valid'.format(hp['logs_prefix'])

    if hp['overwrite_logs']:
        print("removing " + train_logs_dir)
        print("removing " + valid_logs_dir)
        try:
            shutil.rmtree(train_logs_dir)
            shutil.rmtree(valid_logs_dir)
            time.sleep(5)
        except FileNotFoundError:
            pass

    with graph.as_default():
        saver = tf.train.Saver()
        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(train_logs_dir, graph=tf.get_default_graph())
        valid_writer = tf.summary.FileWriter(valid_logs_dir, graph=tf.get_default_graph())

    with tf.Session(graph=graph) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        step = 0
        while True:
            step += 1
            batch_x, batch_y = sess.run([X_train_batch_op, y_train_batch_op])
            summary, _ = sess.run([merged_summary, training_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: hp['keep_prob']})

            if step % hp['eval_frequency'] == 0:
                train_writer.add_summary(summary, step)

                saver.save(sess, hp['logs_dir'] + '/{}-step{}.ckpt'.format(hp['model'].__name__, step))

                batch_x, batch_y = sess.run([X_test_batch_op, y_test_batch_op])
                summary, _, kp = sess.run([merged_summary, accuracy_operation, keep_prob], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                valid_writer.add_summary(summary, step)

        coord.request_stop()
        coord.join(threads)

        saver.save(sess, hp['logs_dir'] + '/{}.ckpt'.format(hp['model'].__name__))
        print("Model saved")

def main():
    hyperparameters_3layer = {
        'model': StrommeNet3Layer,
        'epochs': 100,
        'image_width':IMAGE_WIDTH,
        'image_height':IMAGE_HEIGHT,
        'pixel_depth': 255,
        'eval_frequency' : 100,
        'batch_size': 8,
        'learning_rate': 0.0005,
        'num_channels': 1,
        'keep_prob': 0.6,
        'conv1_num_outputs' : 64,
        'conv2_num_outputs': 128,
        'conv3_num_outputs': 256,
        'logs_dir': './logs',
        'overwrite_logs': True,
        'logs_prefix': 'StrommeNet3Layer', #'{}'.format(run_num), #datetime.now().strftime('%Y-%m-%d--%H.%M.%S'),
        'train_filename': "hwdb1.1.train.tfrecords",
        'test_filename': "hwdb1.1.test.tfrecords",
        'n_classes': len(labels),
    }

    train(hyperparameters_3layer)

    # train_size = 784907
    # test_size = 336842

if __name__ == '__main__':
    main()
