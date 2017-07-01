from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)

#Params
lr=0.001
training_iters=100000
batch_size=128
display_step=10

n_input=28
n_steps=28
n_hidden=128
n_classes=10

x=tf.placeholder("float",[None,n_steps,n_input])
y=tf.placeholder("float",[None,n_classes])

weights={
	"out":tf.Variable(tf.random_normal([2*n_hidden,n_classes]))
}
biases={
	"out":tf.Variable(tf.random_normal([n_classes]))
}

def BiRNN(x,weights,biases):
	x=tf.unstack(x,n_steps,1)
	lstm_fw_cell=rnn.BasicLSTMCell(n_hidden)
	lstm_bw_cell=rnn.BasicLSTMCell(n_hidden)
	outputs,_,_=rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,dtype=tf.float32)
	return tf.matmul(outputs[-1],weights["out"])+biases["out"]

pred=BiRNN(x,weights,biases)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#Initialize
init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	step=1
	while step*batch_size<training_iters:
		batch_x,batch_y=mnist.train.next_batch(batch_size)
		batch_x=batch_x.reshape((batch_size,n_steps,n_input))
		sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
		if step%display_step==0:
			acc,loss=sess.run([accuracy,cost],feed_dict={x:batch_x,y:batch_y})
			print("Iters "+str(step*batch_size)+",loss = "+"{:.6f}".format(loss)+",acc = "+"{:.6f}".format(acc))
		step+=1
	print ("Optimization Finished!")
	test_data=mnist.test.images[:batch_size]
	test_labels=mnist.test.labels[:batch_size]
	test_data=test_data.reshape((batch_size,n_steps,n_input))
	print("Testing Accuracy = ",sess.run(accuracy,feed_dict={x:test_data,y:test_labels}))
