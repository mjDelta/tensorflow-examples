# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:27:57 2017

@author: ZMJ
"""
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

lr=0.01
batch_size=128
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
logs_path="/tmp/tensorflow_logs/example"

n_input=28*28
n_classes=10
n_hidden_1=256
n_hidden_2=128

x=tf.placeholder(tf.float32,[None,n_input],name="InputData")
y=tf.placeholder(tf.float32,[None,n_classes],name="LabelData")

weights={
    "layer1":tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    "layer2":tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    "out":tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
    }
biases={
    "layer1":tf.Variable(tf.random_normal([n_hidden_1])),
    "layer2":tf.Variable(tf.random_normal([n_hidden_2])),
    "out":tf.Variable(tf.random_normal([n_classes]))
    }

def multiPerceptron(x,weights,biases):
  with tf.name_scope("Model"):
    layer1=tf.nn.relu(tf.matmul(x,weights["layer1"])+biases["layer1"])
    layer2=tf.nn.relu(tf.matmul(layer1,weights["layer2"])+biases["layer2"])
    return tf.matmul(layer2,weights["out"])+biases["out"]

pred=multiPerceptron(x,weights,biases)

with tf.name_scope("Loss"):
  cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
with tf.name_scope("Optimizer"):
  optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
with tf.name_scope("Accurcy"):
  correct=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
  acc=tf.cast(correct,"float")

init=tf.global_variables_initializer()

tf.summary.scalar("loss",cost)
#tf.summary.scalar("accuracy",acc)
merged_summary_op=tf.summary.merge_all()

with tf.Session() as sess:
  sess.run(init) 
  summary_writer=tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())
  num_batch=int(mnist.train.num_examples/batch_size)
  for epoch in range(7):
    for i in range(num_batch):
      batch_x,batch_y=mnist.train.next_batch(batch_size)
      _,summary=sess.run([optimizer,merged_summary_op],feed_dict={x:batch_x,y:batch_y})
      summary_writer.add_summary(summary,epoch*num_batch+i)

  print("Run the command line:\n" \
          "--> tensorboard --logdir=/tmp/tensorflow_logs " \
"\nThen open http://0.0.0.0:6006/ into your web browser")      
      
