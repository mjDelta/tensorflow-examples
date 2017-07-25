# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:40:48 2017

@author: ZMJ
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
lr=0.01

n_input=28*28
n_classes=10
batch_size=128

epochs=20
display_step=1

x=tf.placeholder("float",[None,n_input])
y=tf.placeholder("float",[None,n_classes])

weight=tf.Variable(tf.random_normal([n_input,n_classes]))
bias=tf.Variable(tf.random_normal([n_classes]))

"""
Model
"""
pred=tf.matmul(x,weight)+bias
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,"float"))

init=tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  batch_total=int(mnist.train.num_examples/batch_size)
  for epoch in range(epochs):
    for i in range(batch_total):
      batch_x,batch_y=mnist.train.next_batch(batch_size)
      sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
    if epoch%display_step==0:
      acc,c=sess.run([accuracy,cost],feed_dict={x:batch_x,y:batch_y})
      print("Epoch "+str(epoch)+". Cost= "+str(c)+".Accuracy ="+str(acc))
  print("Optimization Finished!")
  acc,c=sess.run([accuracy,cost],feed_dict={x:mnist.test.images,y:mnist.test.labels})
  print("Testing Cost ="+str(c)+".Testing Accuracy="+str(acc))