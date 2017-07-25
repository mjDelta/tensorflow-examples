# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 20:27:50 2017

@author: ZMJ
"""
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

lr=0.001
batch_size=100
display_step=1
model_path="/tmp/model.ckpt"

n_hidden_1=256
n_hidden_2=256
n_input=784
n_classes=10

x=tf.placeholder("float",[None,n_input])
y=tf.placeholder("float",[None,n_classes])

weights={
    "hidden_1":tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    "hidden_2":tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    "out":tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
    }

biases={
    "hidden_1":tf.Variable(tf.random_normal([n_hidden_1])),
    "hidden_2":tf.Variable(tf.random_normal([n_hidden_2])),
    "out":tf.Variable(tf.random_normal([n_classes]))
    }

def multilayer_perceptron(x,weights,biases):
  layer_1=tf.nn.relu(tf.matmul(x,weights["hidden_1"])+biases["hidden_1"])
  layer_2=tf.nn.relu(tf.matmul(layer_1,weights["hidden_2"])+biases["hidden_2"])
  out_layer=tf.matmul(layer_2,weights["out"])+biases["out"]
  return out_layer

pred=multilayer_perceptron(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

init=tf.global_variables_initializer()

"""
Saver Define
"""
saver=tf.train.Saver()

print("Startint 1st session...")
with tf.Session() as sess:
  sess.run(init)
  for epoch in range(3):
    total_batch=int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
      batch_x,batch_y=mnist.train.next_batch(batch_size)
      sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
    if epoch%1==0:
      loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
      print("Epoch "+str(epoch)+".Training Cost="+str(loss))
  print("First Optimization Finished!")
  
  correct=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
  acc=tf.reduce_mean(tf.cast(correct,"float"))
  print("Accuracy :",acc.eval({x:mnist.test.images,y:mnist.test.labels}))

  """
  Save Model
  """
  save_path=saver.save(sess,model_path)
  print("Model saved in file:"+str(save_path))
  
  print("Starting 2nd session...")
  with tf.Session() as sess:
    sess.run(init)
    
    """
    Restore Model
    """
    saver.restore(sess,model_path)
    print("Model restored from file:"+str(save_path))
    
    for epoch in range(7):
      total_batch=int(mnist.train.num_examples/batch_size)
      for i in range(total_batch):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
      if epoch%display_step==0:
        loss=sess.run(cost,feed_dict={x:batch_x,y:batch_y})
        print("Epoch "+str(epoch)+".Training Cost="+str(loss))
        
    print("Second Optimization Finisehed!")
    
    correct=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    acc=tf.reduce_mean(tf.cast(correct,"float"))
    
    print("Accuracy ",acc.eval({x:mnist.test.images,y:mnist.test.labels}))
    
  
  
  
  
  
