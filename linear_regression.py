# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 22:59:41 2017

@author: ZMJ
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print "Package Loaded"

np.random.seed(1)
def f(x,weight,bias):
  return x*weight+bias

Wref=0.7
Bref=-0.1
n=20
noise_var=0.05
train_X=np.random.random((n,1))
ref_Y=f(train_X,Wref,Bref)
train_Y=ref_Y+noise_var*np.random.randn(n,1)

lr=0.01
epochs=10000
display_step=50

n_samples=train_X.size
plt.figure(1)
plt.axis("equal")
plt.plot(train_X[:,0],ref_Y[:,0],"ro",label="Original Data")
plt.plot(train_X[:,0],train_Y[:,0],"bo",label="Training Data")
plt.title("Sactter Plot of Data")
plt.legend(loc="lower right")

weight=tf.Variable(np.random.randn(),name="weight")
bias=tf.Variable(np.random.randn(),name="bias")
x=tf.placeholder(tf.float32,shape=[n_samples,1],name="input")
y=tf.placeholder(tf.float32,shape=[n_samples,1],name="output")

"""
Model
"""
pred=x*weight+bias
cost=tf.reduce_mean(tf.pow(pred-y,2))
optimizer=tf.train.GradientDescentOptimizer(lr).minimize(cost)
init=tf.global_variables_initializer()

"""
Run Model in Session
"""
with tf.Session() as sess:
  sess.run(init)
  for epoch in range(epochs):
    l=sess.run(optimizer,feed_dict={x:train_X,y:train_Y})
    if epoch%display_step==0:
      c=sess.run(cost,feed_dict={x:train_X,y:train_Y})
      print "Epoch %s .Cost=%s"%(epoch,c)
  Wop=sess.run(weight)
  Bop=sess.run(bias)
  fop=f(train_X,Wop,Bop)      
  plt.figure(2)
  plt.plot()
  plt.plot(train_X[:,0],ref_Y[:,0],"ro",label="Original Data")
  plt.plot(train_X[:,0],train_Y[:,0],"bo",label="Training Data")
  plt.plot(train_X[:,0],fop[:,0],"k-",label="Predicted Line")
  plt.title("Predicted Line")
  plt.legend(loc="lower right")
  plt.show()
