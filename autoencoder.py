# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 12:23:32 2017

@author: ZMJ
"""

from __future__ import division,print_function,absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

"""
Model Defination
"""
lr=0.01
training_epochs=20
batch_size=256
display_step=1
examples_to_show=10
n_hidden1=256
n_hidden2=128
n_input=784

X=tf.placeholder(tf.float32,[None,n_input])

weights={
    "encoder_h1":tf.Variable(tf.random_normal([n_input,n_hidden1])),
    "encoder_h2":tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),
    "decoder_h1":tf.Variable(tf.random_normal([n_hidden2,n_hidden1])),
    "decoder_h2":tf.Variable(tf.random_normal([n_hidden1,n_input]))
}

biases={
    "encoder_h1":tf.Variable(tf.random_normal([n_hidden1])),
    "encoder_h2":tf.Variable(tf.random_normal([n_hidden2])),
    "decoder_h1":tf.Variable(tf.random_normal([n_hidden1])),
    "decoder_h2":tf.Variable(tf.random_normal([n_input]))
}

def encoder(x):
  layer1=tf.nn.sigmoid(tf.matmul(x,weights["encoder_h1"])+biases["encoder_h1"])
  layer2=tf.nn.sigmoid(tf.matmul(layer1,weights["encoder_h2"])+biases["encoder_h2"])
  return layer2

def decoder(x):
  layer1=tf.nn.sigmoid(tf.matmul(x,weights["decoder_h1"])+biases["decoder_h1"])
  layer2=tf.nn.sigmoid(tf.matmul(layer1,weights["decoder_h2"])+biases["decoder_h2"])
  return layer2

"""
Model Construction
"""
encoder_op=encoder(X)
decoder_op=decoder(encoder_op)
y_pred=decoder_op
y_true=X

cost=tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer=tf.train.RMSPropOptimizer(lr).minimize(cost)

init=tf.global_variables_initializer()


"""
Launch the Graph
"""
with tf.Session() as sess:
  sess.run(init)
  total_batch=int(mnist.train.num_examples/batch_size)
  for epoch in range(training_epochs):
    for i in range(total_batch):
      batch_x,batch_y=mnist.train.next_batch(batch_size)
      sess.run(optimizer,feed_dict={X:batch_x})
    if epoch%display_step==0:
      loss=sess.run(cost,feed_dict={X:batch_x})
      print("Epoch "+str(epoch)+".Cost = "+"{:.9f}".format(loss))
  print("Optimizer Finished!")
  y_pred=sess.run(y_pred,feed_dict={X:mnist.test.images[:examples_to_show]})

  for i in range(examples_to_show):
    plt.subplot(121)
    plt.imshow(np.reshape(mnist.test.images[i],(28,28)))
    plt.subplot(122)
    plt.imshow(np.reshape(y_pred[i],(28,28)))
    plt.show()
  
