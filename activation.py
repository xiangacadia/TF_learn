import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def fq(x, min_value, max_value, num_intervals):
  # calculate interval size
  interval = (max_value - min_value) / float(num_intervals)
  # calculate embedding id
  embed_id = tf.to_int32(tf.floordiv(tf.sub(x, min_value), interval))
  embed_id = tf.cond(tf.squeeze(embed_id) > 9, lambda: tf.sub(embed_id, 1), lambda: embed_id)
  #embed_id = tf.maximum(embed_id, 9)
  embeddings = tf.Variable(
    tf.random_uniform([num_intervals, 1], -1.0, 1.0))
  # Look up embeddings for parameters
  embed = tf.nn.embedding_lookup(embeddings, embed_id)
  y = tf.squeeze(embed)
  return y

def create_hashing(x, min_value, max_value, num_intervals):
  num_features = int(x.get_shape()[0])
  new_features = []
  for i in range(num_features):
    feature = tf.slice(x, [i], [1])
    y = fq(feature, min_value, max_value, num_intervals)
    new_features.append(y)
  new_features = tf.pack(new_features)
  return new_features
    

def test():
  # Import MNIST data
  mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

  num_intervals = 10
  num_features = 784
  num_nodes = 10

  x = tf.placeholder(tf.float32, [num_features])
  label = tf.placeholder(tf.float32)
  hashed_features = create_hashing(x, 0, 1, num_intervals)
  hashed_features = tf.reshape(hashed_features, [1, num_features])

  # add hidden layer
  W = tf.Variable(tf.zeros([num_features, num_nodes]))
  b = tf.Variable(tf.zeros([num_nodes]))

  # output layer
  pre_activation = tf.matmul(hashed_features, W) + b
  y = tf.nn.softmax(pre_activation) # Softmax

  loss = tf.reduce_mean(tf.square((y - label)))
  optimizer = tf.train.GradientDescentOptimizer(0.1)
  train = optimizer.minimize(loss)

  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)

  for step in xrange(2):
    print step
    for i in range(10):
      batch_x, batch_label = mnist.train.next_batch(1)
      batch_x = batch_x.reshape(784)
      print batch_label
      sess.run(train, feed_dict={x: batch_x, label: batch_label})
      pred =  sess.run(y, feed_dict={x: batch_x, label: batch_label})
      print pred
  batch_x, batch_label = mnist.train.next_batch(100)
  print sess.run(y, feed_dict={x: batch_x})

  print data_y
    

if __name__ == "__main__":
  test()
