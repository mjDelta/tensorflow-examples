# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 13:57:25 2017

@author: ZMJ
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("data/",one_hot=True)

trainimg=mnist.train.images
trainlabel=mnist.train.labels
testimg=mnist.test.images
testlabel=mnist.test.labels

"""
Network Params Define
"""
n_input=784
n_hidden_1=256
n_hidden_2=256
n_output=784

x=tf.placeholder("float",[None,n_input])
y=tf.placeholder("float",[None,n_output])
keep_prob=tf.placeholder("float")

weights={
    "h1":tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    "h2":tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    "out":tf.Variable(tf.random_normal([n_hidden_2,n_output]))
    }

biases={
    "h1":tf.Variable(tf.random_normal([n_hidden_1])),
    "h2":tf.Variable(tf.random_normal([n_hidden_2])),
    "out":tf.Variable(tf.random_normal([n_output]))
    }

def dae(x,weights,biases,keep_prob):
  layer_1=tf.nn.sigmoid(tf.matmul(x,weights["h1"])+biases["h1"])
  layer_1_drop=tf.nn.dropout(layer_1,keep_prob)
  layer_2=tf.nn.sigmoid(tf.matmul(layer_1_drop,weights["h2"])+biases["h2"])
  layer_2_drop=tf.nn.dropout(layer_2,keep_prob)
  return tf.sigmoid(tf.matmul(layer_2_drop,weights["out"])+biases["out"])

recon=dae(x,weights,biases,keep_prob)
cost=tf.reduce_mean(tf.pow(recon-y,2))
optimizer=tf.train.AdamOptimizer(0.01).minimize(cost)
init=tf.global_variables_initializer()

"""
Saver Difine
"""
savedir="nets/"
saver=tf.train.Saver(max_to_keep=1)

epochs=41
batch_size=100
display_step=5

with tf.Session() as sess:
  sess.run(init)
  num_batch=int(mnist.train.num_examples/batch_size)
  randidx=np.random.randint(testimg.shape[0],size=1)
  for epoch in range(epochs):
    for i in range(num_batch):
      batch_x,batch_y=mnist.train.next_batch(batch_size)
      batch_x_noise=batch_x+0.3*np.random.randn(batch_size,n_input)

      sess.run(optimizer,feed_dict={x:batch_x_noise,y:batch_x,keep_prob:1.})
      if(i%100==0):
        print("Epoch %s i %s:Loss = %s"%(epoch,i,sess.run(cost,feed_dict={x:batch_x_noise,y:batch_x,keep_prob:1.0})))
    if epoch%display_step==0:
      print("Epoch %s:Loss = %s"%(epoch,sess.run(cost,feed_dict={x:batch_x_noise,y:batch_x,keep_prob:1.0})))
      
      testvec=testimg[randidx,:]
      noisevec=testvec+0.3*np.random.randn(1,n_input)
      outvec=sess.run(recon,feed_dict={x:noisevec,y:testvec,keep_prob:1.})
      outimg=np.reshape(outvec,(28,28))
      
      """
      Plot
      """
      plt.subplot(131)
      plt.imshow(np.reshape(testvec,(28,28)),cmap=plt.get_cmap("gray"))
      plt.title("Epoch %s/%s:Origin Image"%(epoch,epochs))


      plt.subplot(132)
      plt.imshow(np.reshape(noisevec,(28,28)),cmap=plt.get_cmap("gray"))
      plt.title("Epoch %s/%s:Noise Image"%(epoch,epochs))

      
      plt.subplot(133)
      plt.imshow(outimg,cmap=plt.get_cmap("gray"))
      plt.title("Epoch %s/%s:Reconstruct Image"%(epoch,epochs))

      plt.show()
    saver.save(sess,savedir+"dae.ckpt",global_step=epoch)
      


  
