import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("data",one_hot=True)
img_h=28
img_w=28
img_d=1
noise_dim=100
input_real=tf.placeholder(tf.float32,[None,img_h,img_w,img_d],name="input_real")
input_noise=tf.placeholder(tf.float32,[None,noise_dim],name="input_noise")

def get_generator(noise_dim,output_dim,is_train=True,alpha=0.01):
	with tf.variable_scope("generator",reuse=(not is_train)):
		layer1=tf.layers.dense(noise_dim,4*4*512)##100-->4*4*512
		layer1=tf.reshape(layer1,[-1,4,4,512])
		layer1=tf.layers.batch_normalization(layer1,training=is_train)
		##4*4*512-->7*7*256
		#print layer1.get_shape().as_list()[-1]
		layer2 = tf.layers.conv2d_transpose(layer1, 256, 4, strides=1, padding='valid')
		#layer2=tf.nn.conv2d_transpose(layer1,filter=[4,4,256,512],output_shape=[batch_size,7,7,256],strides=[1,1,1,1],padding="VALID")
		layer2=tf.layers.batch_normalization(layer2,training=is_train)
		layer2=tf.nn.relu(layer2)
		layer2=tf.nn.dropout(layer2,keep_prob=0.8)

		##7*7*256-->14*14*128
		layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides=2, padding='same')

		#layer3=tf.nn.conv2d_transpose(layer2,[3,3,128,256],[batch_size,14,14,128],strides=[1,2,2,1],padding="SAME")
		layer3=tf.layers.batch_normalization(layer3,training=is_train)
		layer3=tf.nn.relu(layer3)
		layer3=tf.nn.dropout(layer3,keep_prob=0.8)

		##14*14*128-->28*28*1
		layer4 = tf.layers.conv2d_transpose(layer3, 1, 3, strides=2, padding='same')
		#layer4=tf.nn.conv2d_transpose(layer3,[3,3,1,128],[batch_size,28,28,1],strides=[1,2,2,1],padding="SAME")

	return tf.nn.sigmoid(layer4)
def get_discriminator(input_img,reuse=False,alpha=0.01):
	with tf.variable_scope("discriminator",reuse=reuse):
		#layer1=tf.layers.batch_normalization(input_img,training=True)
		##28*28*1-->14*14*128
		#print layer1.shape
		layer1 = tf.layers.conv2d(layer1, 128, 3, strides=2, padding='same')
		#layer1=tf.nn.conv2d(layer1,[3,3,1,128],[1,2,2,1],padding="SAME")
		#layer1=tf.layers.batch_normalization(layer1,training=True)
		layer1=tf.maximum(alpha*layer1,layer1)
		layer1=tf.nn.dropout(layer1,keep_prob=0.8)

		##14*14*128-->7*7*256
		layer2 = tf.layers.conv2d(layer1, 256, 3, strides=2, padding='same')
		#layer2=tf.nn.conv2d(layer1,[3,3,128,256],[1,2,2,1],padding="SAME")
		layer2=tf.layers.batch_normalization(layer2,training=True)
		layer2=tf.maximum(alpha*layer2,layer2)
		layer2=tf.nn.dropout(layer2,keep_prob=0.8)

		##7*7*256-->4*4*512
		layer3 = tf.layers.conv2d(layer2, 512, 3, strides=2, padding='same')
		#layer3=tf.nn.conv2d(layer2,[3,3,256,512],[1,2,2,1],padding="SAME")
		layer3=tf.layers.batch_normalization(layer3,training=True)		
		layer3=tf.maximum(alpha*layer3,layer3)
		layer3=tf.nn.dropout(layer3,keep_prob=0.8)

		##4*4*512-->10
		layer4=tf.reshape(layer3,(-1,4*4*512))
		layer4=tf.layers.dense(layer4,10)
	return tf.nn.sigmoid(layer4)
def get_loss(input_real,input_fake,img_d):
	g_imgs=get_generator(input_fake,img_d,is_train=True)
	d_real=get_discriminator(input_real)
	d_fake=get_discriminator(g_imgs,reuse=True)
	
	g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,labels=tf.ones_like(d_fake)))
	d_real_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real,labels=tf.ones_like(d_real)))
	d_fake_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,labels=tf.zeros_like(d_fake)))
	d_loss=d_real_loss+d_fake_loss
	return g_loss,d_loss
def get_optimizer(g_loss,d_loss,beta1=0.4):
	train_vars=tf.trainable_variables()
	g_vars=[var for var in train_vars if var.name.startswith("generator")]
	d_vars=[var for vat in train_vars if var.name.startswith("discriminator")]
	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		g_op=tf.train.AdamOptimizer(beta1=beta1).minimize(g_loss,var_list=g_vars)
		d_op=tf.train.AdamOptimizer(beta1=beta1).minimize(d_loss,var_list=d_vars)
	return g_op,d_op
def plot_imgs(samples):
	fig,axes=plt.subplots(2,10,figsize=(20,4))
	#print samples.shape
	for i in range(10):
		img1=samples[i*2]
		img2=samples[i*2+1]
		axes[0][i].matshow(np.reshape(img1,(28,28)),cmap=plt.get_cmap("gray"))
		axes[1][i].matshow(np.reshape(img2,(28,28)),cmap=plt.get_cmap("gray"))
	fig.tight_layout(pad=0)
	plt.show()
def show_generator_output(sess,n_images,inputs_noise,output_dim):
	cmap="Greys_r"
	noise_shape=inputs_noise.get_shape().as_list()[-1]
	example_noise=np.random.uniform(-1,1,size=[n_images,noise_shape])
	samples=sess.run(get_generator(inputs_noise,output_dim,False),feed_dict={inputs_noise:example_noise})
	result=np.squeeze(samples,-1)
	return result

batch_size=64
noise_size=100
epochs=5
n_samples=20
beta1=0.4


losses=[]
step=0
g_loss,d_loss=get_loss(input_real,input_noise,1)
g_train_op,d_train_op=get_optimizer(g_loss,d_loss,beta1)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		for batch_i in range(mnist.train.num_examples//batch_size):
			step+=1
			batch_x=mnist.train.next_batch(batch_size)
			batch_imgs=batch_x[0].reshape((batch_size,28,28,1))
			batch_imgs=batch_imgs*2-1
			batch_noise=np.random.uniform(-1,1,size=(batch_size,noise_dim))
			sess.run(g_train_op,feed_dict={input_real:batch_imgs,input_noise:batch_noise})
			sess.run(d_train_op,feed_dict={input_real:batch_imgs,input_noise:batch_noise})
			if step%101==0:
				train_g_loss=g_loss.eval({input_real:batch_imgs,input_noise:batch_noise})
				train_d_loss=d_loss.eval({input_real:batch_imgs,input_noise:batch_noise})
				losses.append((train_g_loss,train_d_loss))
				samples=show_generator_output(sess,n_samples,input_noise,1)
				plot_imgs(samples)
				print "Epoch %s/%s:Discriminator Loss:%s.Generator Loss:%s"%(epoch,epochs,train_d_loss,train_g_loss)

					
	samples=show_generator_output(sess,n_samples,input_noise,1)
	plot_imgs(samples)	
	print "Epoch %s/%s:Discriminator Loss:%s.Generator Loss:%s"%(epoch,epochs,train_d_loss,train_g_loss)

