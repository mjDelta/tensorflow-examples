from __future__ import print_function

#import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)

import tensorflow as tf

lr=0.001
training_epochs=20
batch_size=100
display_step=1


#Network Parameters
n_hidden_1=512 #1st layer number of features
n_hidden_2=512#2nd layer number of features
n_hidden_3=256
n_input=784#MNIST data input(img shape:28*28)
n_classes=10#MNIST total classes(0-9 digits)

#tf Graph input
x=tf.placeholder("float",[None,n_input])
y=tf.placeholder("float",[None,n_classes])
keep_prob=tf.placeholder("float")
stddev=0.05
#store layers weights & biases
weights={
	"h1":tf.Variable(tf.random_normal([n_input,n_hidden_1],stddev=stddev)),
	"h2":tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2],stddev=stddev)),
  "h3":tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3],stddev=stddev)),
	"out":tf.Variable(tf.random_normal([n_hidden_3,n_classes],stddev=stddev))
}
biases={
	"b1":tf.Variable(tf.random_normal([n_hidden_1])),
	"b2":tf.Variable(tf.random_normal([n_hidden_2])),
  "b3":tf.Variable(tf.random_normal([n_hidden_3])),
	"out":tf.Variable(tf.random_normal([n_classes]))
}
#create model
def multilayer_perceptron(x,weights,biases):
  layer1=tf.nn.relu(tf.matmul(x,weights['h1'])+biases['b1'])
  layer2=tf.nn.relu(tf.matmul(layer1,weights["h2"])+biases["b2"])
  layer3=tf.nn.relu(tf.matmul(layer2,weights["h3"])+biases["b3"])
  layer3=tf.nn.dropout(layer3,keep_prob)
  out_layer=tf.matmul(layer3,weights["out"])+biases["out"]
  return out_layer
#construct model
pred=multilayer_perceptron(x,weights,biases)
#define loss and optimizer
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
#initialize the variables
init=tf.global_variables_initializer()
correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
#launch the graph
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(training_epochs):

		total_batch=(int)(mnist.train.num_examples/batch_size)
		for i in range(total_batch):
			batch_x,batch_y=mnist.train.next_batch(batch_size)
			_,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y,keep_prob:0.6})

		if epoch%display_step==0:
			c=sess.run(cost,feed_dict={x:batch_x,y:batch_y,keep_prob:1.0})
			print ("Epoch:","%04d"%(epoch+1),"cost=","{:.9f}".format(c))
	print ("Optimization Finished!")
	#Test model

	print ("Accuracy:",accuracy.eval({x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0}))


