# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:20:18 2017

@author: ZMJ
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

lr=0.01
training_epochs=25
batch_size=100
display_step=1
logs_path="/tmp/tensorflow_logs/example"

n_hidden_1=256
n_hidden_2=256
n_input=28*28
n_classes=10

x=tf.placeholder("float",[None,n_input],name="InputData")
y=tf.placeholder("float",[None,n_classes],name="LabelData")

weights={
    "w1":tf.Variable(tf.random_normal([n_input,n_hidden_1]),name="w1"),
    "w2":tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2]),name="w2"),
    "out":tf.Variable(tf.random_normal([n_hidden_2,n_classes]),name="wout")
    }

biases={
    "b1":tf.Variable(tf.random_normal([n_hidden_1]),name="b1"),
    "b2":tf.Variable(tf.random_normal([n_hidden_2]),name="b2"),
    "out":tf.Variable(tf.random_normal([n_classes]),name="bout")
    }

def multiPerceptron(x,weights,biases):
  layer1=tf.nn.relu(tf.matmul(x,weights["w1"])+biases["b1"])
  #tf.summary.histogram("relu1",layer1)
  layer2=tf.nn.relu(tf.matmul(layer1,weights["w2"])+biases["b2"])
  #tf.summary.histogram("relu2",layer2)  
  return tf.matmul(layer2,weights["out"])+biases["out"]

"""
Making TensorBoard's Graph
"""
with tf.name_scope("Model"):
  pred=multiPerceptron(x,weights,biases)
  
with tf.name_scope("Loss"):
  loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
  
with tf.name_scope("AdamOptimizer"):
  optimizer=tf.train.AdamOptimizer(lr).minimize(loss)


with tf.name_scope("Accuracy"):
  acc=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
  acc=tf.reduce_mean(tf.cast(acc,tf.float32))

init=tf.global_variables_initializer()

tf.summary.scalar("loss",loss)
tf.summary.scalar("accuracy",acc)

##visulize weights summary 
#for var in tf.trainable_variables():
#  tf.summary.histogram(var.name,var)

##visulize gradients summary
#for grad,var in grads:
#  tf.summary.histogram(var.name+"/gradients",grad)
  
merged_summary_op=tf.summary.merge_all()

with tf.Session() as sess:
  sess.run(init)
  
  ##Write logs to tensorboard
  summary_writer=tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())
  
  for epoch in range(training_epochs):
    avg_cost=0.
    batch_num=int(mnist.train.num_examples/batch_size)
    for i in range(batch_num):
      batch_x,batch_y=mnist.train.next_batch(batch_size)
      _,c,summary=sess.run([optimizer,loss,merged_summary_op],feed_dict={x:batch_x,y:batch_y})
      summary_writer.add_summary(summary,epoch*batch_num+i)
      avg_cost+=c/batch_num
    if (epoch+1)%display_step==0:
      print("Epoch "+str(epoch)+".Cost= "+str(c))
  print("Optimization Finished!")
  print("Accuracy ",acc.eval({x:mnist.test.images,y:mnist.test.labels}))
  
  print("Run the command line:\ntensorboard --logdir=/tmp/tensorflow_logs\nThen open http://0.0.0.0:6006")
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
