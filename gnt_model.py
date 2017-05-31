import tensorflow as tf
from utils.gnt_record import read_and_decode, BATCH_SIZE


with open('label_keys.list') as f:
    labels = f.readlines()

tfrecords_filename = "hwdb1.1.tfrecords"
filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=10)

# Even when reading in multiple threads, share the filename
# queue.
images_batch, labels_batch = read_and_decode(filename_queue)
label_one_hot = tf.one_hot(labels_batch, len(labels))

print(label_one_hot)

# simple model
images_batch_normalized = images_batch / 128 - 0.5
print(images_batch)
print(images_batch_normalized)

images_batch_normalized = tf.reshape(images_batch_normalized, [BATCH_SIZE, 128*128])
print(images_batch_normalized)

w = tf.get_variable("w1", [128*128, len(labels)])
y_pred = tf.matmul(images_batch_normalized, w)

print("y pred & labels batch")
print(y_pred)
print(label_one_hot)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_batch, logits=y_pred)

# for monitoring
loss_mean = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer().minimize(loss)

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session()  as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    while True:
      _, loss_val = sess.run([train_op, loss_mean])
      print(loss_val)

    coord.request_stop()
    coord.join(threads)
