from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)

print(mnist.test.images[0:1])
#Params
lr=0.001
training_iters=10000
batch_size=128
display_step=10
model_path="/tmp/conv.ckpt"

#Network params
n_input=784
n_classes=10
dropout=0.75

x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])
keep_prob=tf.placeholder(tf.float32)
weights={
	"wc1":tf.Variable(tf.random_normal([5,5,1,32])),
	"wc2":tf.Variable(tf.random_normal([5,5,32,64])),
	"wf1":tf.Variable(tf.random_normal([7*7*64,1024])),
	"out":tf.Variable(tf.random_normal([1024,n_classes]))
	}
biases={
	"bc1":tf.Variable(tf.random_normal([32])),
	"bc2":tf.Variable(tf.random_normal([64])),
	"bf1":tf.Variable(tf.random_normal([1024])),
	"out":tf.Variable(tf.random_normal([n_classes]))
}
def conv2d(x,W,b,strides=1):
	x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding="SAME")
	x=tf.nn.bias_add(x,b)
	return tf.nn.relu(x)

def maxpool2d(x,k=2):
	return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding="SAME")

def conv_net(x,weights,biases,dropout):
  x=tf.reshape(x,shape=[-1,28,28,1])
  conv1=conv2d(x,weights["wc1"],biases["bc1"])
  pool1=maxpool2d(conv1,k=2)
	
  conv2=conv2d(pool1,weights["wc2"],biases["bc2"])
  pool2=maxpool2d(conv2,k=2)
	
  fc1=tf.reshape(pool2,[-1,weights["wf1"].get_shape().as_list()[0]])
  fc1=tf.matmul(fc1,weights["wf1"])+biases["bf1"]
  relu1=tf.nn.relu(fc1)
  dropout1=tf.nn.dropout(relu1,dropout)
	
  _out=tf.matmul(dropout1,weights["out"])+biases["out"]
  out={"x":x,"conv1":conv1,"pool1":pool1,"conv2":conv2,"pool2":pool2,"fc1":fc1,"relu1":relu1,"dropout1":dropout1,"out":_out}
  return out

pred=conv_net(x,weights,biases,keep_prob)["out"]

#logits=tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y)
logits=tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred)
cost=tf.reduce_mean(logits)
optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()

saver=tf.train.Saver()
with tf.Session() as sess:
  sess.run(init)
  step=1
  while step*batch_size<training_iters:
    batch_x,batch_y=mnist.train.next_batch(batch_size)

    sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:dropout})
    if step%display_step==0:
      loss,acc=sess.run([cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob:1.})
      print("Iter "+str(step*batch_size)+",mini_batch_loss="+"{:.6f}".format(loss)+",accuracy="+"{:.5f}".format(acc))
    step+=1
  print ("Optimization finished!")
  print ("Testing Accuracy:",sess.run(accuracy,feed_dict={x:mnist.test.images[:256],y:mnist.test.labels[:256],keep_prob:1.0}))
  save_path=saver.save(sess,model_path)

with tf.Session() as sess:
  saver.restore(sess,model_path)
  conv_out=conv_net(x,weights,biases,dropout)
  out=sess.run(conv_out,feed_dict={x:mnist.test.images[0:1]})
  x=out["x"]
  conv1=out["conv1"]
  conv2=out["conv2"]
  pool2=out["pool2"]
  fc1=out["fc1"]
  relu1=out["relu1"]
  dropout1=out["dropout1"]
  out=out["out"]
	#x=sess.run(conv_out["x"],feed_dict={x:mnist.test.images[0:1]})
	#conv1=sess.run(conv_out["conv1"],feed_dict={x:mnist.test.images[0:1]})
	#conv2=sess.run(conv_out["conv2"],feed_dict={x:mnist.test.images[0:1]})
	#pool2=sess.run(conv_out["pool2"],feed_dict={x:mnist.test.images[0:1]})
	#fc1=sess.run(conv_out["fc1"],feed_dict={x:mnist.test.images[0:1]})
	#relu1=sess.run(conv_out["relu1"],feed_dict={x:mnist.test.images[0:1]})
	#dropout1=sess.run(conv_out["dropout1"],feed_dict={x:mnist.test.images[0:1]})
	#out=sess.run(conv_out["out"],feed_dict={x:mnist.test.images[0:1]})

  """Process Visualization"""
  plt.matshow(x[0, :, :, 0], cmap=plt.get_cmap('gray'))
  plt.title("Input:Label of this image is "+str(np.argmax(mnist.test.labels[0, :])))
  plt.colorbar()

  for i in range(3):
    
    plt.matshow(conv1[0, :, :, i], cmap=plt.get_cmap('gray'))
    plt.title("conv1:Label of this image is "+str(np.argmax(mnist.test.labels[0, :]))+"channel "+str(i))
    plt.colorbar() 
  
    #plt.matshow(conv2[0, :, :, i], cmap=plt.get_cmap('gray'))
    #plt.title("conv2:Label of this image is "+str(np.argmax(mnist.test.labels[0, :]))+"channel "+str(i))
    #plt.colorbar()  
   
    #plt.matshow(pool2[0, :, :, i], cmap=plt.get_cmap('gray'))
    #plt.title("pool2:Label of this image is "+str(np.argmax(mnist.test.labels[0, :]))+"channel "+str(i))
    #plt.colorbar()
  plt.show()
