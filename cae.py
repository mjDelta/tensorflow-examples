import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist=input_data.read_data_sets("data/",one_hot=True)
trainimgs=mnist.train.images
trainlabels=mnist.train.labels
testimgs=mnist.test.images
testlabels=mnist.test.labels

ntrain=trainimgs.shape[0]
ntest=testimgs.shape[0]
ndim=784


"""
Network Defination
"""
n1=16
n2=32
n3=64
ksize=5
x=tf.placeholder(tf.float32,[None,ndim])
y=tf.placeholder(tf.float32,[None,ndim])
keep_prob=tf.placeholder("float")
weights={
	"ce1":tf.get_variable("ce1",shape=[ksize,ksize,1,n1],initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
	"ce2":tf.get_variable("ce2",shape=[ksize,ksize,n1,n2],initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
	"ce3":tf.get_variable("ce3",shape=[ksize,ksize,n2,n3],initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
	"dd1":tf.get_variable("dd1",shape=[ksize,ksize,n2,n3],initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
	"dd2":tf.get_variable("dd2",shape=[ksize,ksize,n1,n2],initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
	"dd3":tf.get_variable("dd3",shape=[ksize,ksize,1,n1],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
}
biases={
	"be1":tf.Variable(tf.random_normal([n1])),
	"be2":tf.Variable(tf.random_normal([n2])),
	"be3":tf.Variable(tf.random_normal([n3])),
	"bd1":tf.Variable(tf.random_normal([n2])),
	"bd2":tf.Variable(tf.random_normal([n1])),
	"bd3":tf.Variable(tf.random_normal([1]))
}
def cae(x,weights,biases,keep_prob):
	input_img=tf.reshape(x,shape=[-1,28,28,1])
	
	##Encoder
	en_layer1=tf.nn.sigmoid(tf.nn.conv2d(input_img,weights["ce1"],strides=[1,2,2,1],padding="SAME")+biases["be1"])
	en_layer1=tf.nn.dropout(en_layer1,keep_prob)
	en_layer2=tf.nn.sigmoid(tf.nn.conv2d(en_layer1,weights["ce2"],strides=[1,2,2,1],padding="SAME")+biases["be2"])
	en_layer2=tf.nn.dropout(en_layer2,keep_prob)
	en_layer3=tf.nn.sigmoid(tf.nn.conv2d(en_layer2,weights["ce3"],strides=[1,2,2,1],padding="SAME")+biases["be3"])
	en_layer3=tf.nn.dropout(en_layer3,keep_prob)

	##Decoder
	de_layer1=tf.nn.sigmoid(tf.nn.conv2d_transpose(en_layer3,weights["dd1"],\
						[tf.shape(x)[0],7,7,n2],strides=[1,2,2,1],padding="SAME")+biases["bd1"])
	de_layer1=tf.nn.dropout(de_layer1,keep_prob)
	de_layer2=tf.nn.sigmoid(tf.nn.conv2d_transpose(de_layer1,weights["dd2"],\
						[tf.shape(x)[0],14,14,n1],strides=[1,2,2,1],padding="SAME")+biases["bd2"])
	de_layer2=tf.nn.dropout(de_layer2,keep_prob)
	de_layer3=tf.nn.sigmoid(tf.nn.conv2d_transpose(de_layer2,weights["dd3"],\
						[tf.shape(x)[0],28,28,1],strides=[1,2,2,1],padding="SAME")+biases["bd3"])
	de_layer3=tf.nn.dropout(de_layer3,keep_prob)
	return de_layer3
pred=cae(x,weights,biases,keep_prob)
cost=tf.reduce_mean(tf.square(pred-tf.reshape(y,shape=[-1,28,28,1])))
optimizer=tf.train.AdamOptimizer().minimize(cost)
init=tf.global_variables_initializer()

batch_size=128
n_epochs=5
with tf.Session() as sess:
	sess.run(init)
	test_x=testimgs[:5]
	for epoch in range(n_epochs):
		for i in range(ntrain//batch_size):
			train_x,train_y=mnist.train.next_batch(batch_size)
			train_x_noise=train_x+0.3*np.random.randn(np.array(train_x).shape[0],ndim)
			sess.run(optimizer,feed_dict={x:train_x_noise,y:train_x,keep_prob:0.75})
			if (i%100==0):
				print "Epoch %s,minibatch %s,loss is %s"%(epoch,(i+1)*batch_size,sess.run(cost,feed_dict={x:train_x_noise,y:train_x,keep_prob:1}))
		if epoch%1==0:
			test_x_noise=test_x+0.3*np.random.randn(5,ndim)
			recon=sess.run(pred,feed_dict={x:test_x_noise,y:test_x,keep_prob:1})
			fig,axs=plt.subplots(2,5,figsize=(15,4))
			for example in range(5):
				axs[0][example].matshow(np.reshape(test_x_noise[example,:],(28,28)),cmap=plt.get_cmap("gray"))	
				axs[1][example].matshow(np.reshape(test_x[example,:],(28,28)),cmap=plt.get_cmap("gray"))	
			plt.title("%s/%s"%(example,5))			
			plt.show()
