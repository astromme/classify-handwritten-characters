import tensorflow as tf
from utils.gnt_record import read_and_decode, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH
import time
import sys
import numpy

WORK_DIR = 'data'
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
SEED = 66478  # Set to None for random seed.
NUM_EPOCHS = 1000
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
train_size = 784907
test_size = 336842
SUMMARIES_DIR = 'summaries'

def data_type():
    return tf.float32

with open('label_keys.list', encoding='utf8') as f:
    labels = f.readlines()

NUM_LABELS = len(labels)

tfrecords_train_filename = "hwdb1.1.train.tfrecords"
tfrecords_test_filename = "hwdb1.1.test.tfrecords"
train_filename_queue = tf.train.string_input_producer(
    [tfrecords_train_filename], num_epochs=NUM_EPOCHS)

test_filename_queue = tf.train.string_input_producer(
    [tfrecords_train_filename], num_epochs=NUM_EPOCHS)

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

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])

# Even when reading in multiple threads, share the filename
# queue.
images_batch, labels_batch = read_and_decode(train_filename_queue)
test_images_batch, test_labels_batch = read_and_decode(test_filename_queue)

# simple model
with tf.name_scope('train_data'):
    images_batch_normalized = images_batch / PIXEL_DEPTH - 0.5
    #variable_summaries(images_batch_normalized)
test_images_batch_normalized = test_images_batch / PIXEL_DEPTH - 0.5

# The variables below hold all the trainable weights. They are passed an
# initial value which will be assigned when we call:
# {tf.global_variables_initializer().run()}
conv1_weights = tf.Variable(
  tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                      stddev=0.1,
                      seed=SEED, dtype=data_type()))
conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
conv2_weights = tf.Variable(tf.truncated_normal(
  [5, 5, 32, 64], stddev=0.1,
  seed=SEED, dtype=data_type()))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
fc1_weights = tf.Variable(  # fully connected, depth 512.
  tf.truncated_normal([IMAGE_HEIGHT // 4 * IMAGE_WIDTH // 4 * 64, 512],
                      stddev=0.1,
                      seed=SEED,
                      dtype=data_type()))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                            stddev=0.1,
                                            seed=SEED,
                                            dtype=data_type()))
fc2_biases = tf.Variable(tf.constant(
  0.1, shape=[NUM_LABELS], dtype=data_type()))


  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases


# Training computation: logits + cross-entropy loss.
with tf.name_scope('logits'):
    logits = model(images_batch_normalized, train=True)
    variable_summaries(logits)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels_batch, logits=logits))
    tf.summary.scalar('loss', loss)


# L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
              tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
# Add the regularization term to the loss.
loss += 5e-4 * regularizers

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0, dtype=data_type())
# Decay once per epoch, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(
  0.01,                # Base learning rate.
  batch * BATCH_SIZE,  # Current index into the dataset.
  train_size,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)
# Use simple momentum for the optimization.
optimizer = tf.train.MomentumOptimizer(learning_rate,
                                     0.9).minimize(loss,
                                                   global_step=batch)

# Predictions for the current training minibatch.
train_prediction = tf.nn.softmax(logits)

# Predictions for the test and validation, which we'll compute less often.
eval_prediction = tf.nn.softmax(model(test_images_batch_normalized))

saver = tf.train.Saver()

with tf.Session()  as sess:
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/train',
                                          sess.graph)
    test_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/test')

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    start_time = time.time()
    num_steps = NUM_EPOCHS*(train_size//BATCH_SIZE)
    for step in range(num_steps):
        step += 1
        sess.run(optimizer)

        if step % EVAL_FREQUENCY == 0:
            save_path = saver.save(sess, WORK_DIR + "/model-step{}.ckpt".format(step))
            print("Model saved in file: %s" % save_path)

            # fetch some extra nodes' data
            summary, l, lr, labels, predictions = sess.run([merged, loss, learning_rate, labels_batch, train_prediction])
            train_writer.add_summary(summary, step)
            elapsed_time = time.time() - start_time
            start_time = time.time()
            step_time = 1000 * elapsed_time / EVAL_FREQUENCY
            epoch = float(step) * BATCH_SIZE / train_size
            print('Step {} of {} (batch {}), {:.1f} ms per step'.format(step, num_steps, epoch, step_time))
            print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
            print('train error: %.1f%%' % error_rate(predictions, labels))

            summary, test_images, test_labels = sess.run([merged, test_images_batch_normalized, test_labels_batch])
            test_writer.add_summary(summary, step)


            batch_predictions = sess.run(eval_prediction,
                                         feed_dict={test_images_batch_normalized: test_images})


            print('test error: %.1f%%' % error_rate(batch_predictions, test_labels))

            # print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
            # print('Validation error: %.1f%%' % error_rate(
            #     eval_in_batches(validation_data, sess), validation_labels))
            sys.stdout.flush()

    coord.request_stop()
    coord.join(threads)
