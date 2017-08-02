import os
import time
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.misc
from scipy.misc import imread,imresize
from tensorflow.examples.tutorials.mnist import input_data

##Reset existing flag
'''tf.app.flags.FLAGS=tf.python.platform.flags._FlagValues()
tf.app.flags._global_parser=argparse.ArgumentParser()'''
#Set Flags
flags=tf.app.flags
flags.DEFINE_integer("epoch",5,"Epoch to train [5]")
flags.DEFINE_float("learning_rate",0.0002,"Learning rate of Adam [0.0002]")
flags.DEFINE_float("beta1",0.5,"Moment term of adam [0.5]")
flags.DEFINE_integer("train_size",np.inf,"The size of train imgs is [np.inf]")
flags.DEFINE_integer("batch_size",64,"The size of batch imgs [64]")
flags.DEFINE_integer("image_size",108,"The size of image to use(will be center cropped)[108]")
flags.DEFINE_integer("output_size",64,"The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim",3,"Dimension of image color[3]")
flags.DEFINE_string("dataset","mnist","The name of dataset[celebA,mnist,lsun]")
flags.DEFINE_string("checkpoint_dir","checkpoint","Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir","samples","Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_crop",False,"True for training,False for testing[False]")
flags.DEFINE_boolean("visualize",False,"True for visualizing,False for nothing[False]")
flags.DEFINE_boolean("is_train",True,"True for training,False for testing[False]")
FLAGS=flags.FLAGS

###dataset 
mnist=input_data.read_data_sets("data/",one_hot=True)
trainimgs=mnist.train.images
trainlabels=mnist.train.labels

###summary define
image_summary=tf.summary.image
scalar_summary=tf.summary.scalar
histogram_summary=tf.summary.histogram
merge_summary=tf.summary.merge
SummaryWriter=tf.summary.FileWriter

###leaky relu
def lrelu(x,leak=0.2,name="lrelu"):
	return tf.maximum(x,leak*x)

###Batch Normalization

def batch_norm(x,train=True):
	return tf.contrib.layers.batch_norm(x,decay=0.9,updates_collections=None,\
																				epsilon=1e-5,scale=True,is_training=train)
###Affine mapping
def linear(input_,output_size,stddev=0.02,init_bias=0.,with_w=False):
	shape=input_.get_shape().as_list()

	matrix=tf.Variable(tf.random_normal([shape[1],output_size],stddev=stddev))

	#matrix=tf.get_variable("matrix",shape=[shape[1],output_size],initializer=tf.contrib.layers.xavier_initializer(uniform=False))
	bias=tf.Variable(tf.constant(init_bias,shape=[output_size]))
	if with_w:
		return tf.matmul(input_,matrix)+bias,matrix,bias
	else:
		return tf.matmul(input_,matrix)+bias

def conv_cond_concat(x,y):
	x_shape=x.get_shape()
	y_shape=y.get_shape()
	return tf.concat([x,y*tf.ones([x_shape[0],x_shape[1],x_shape[2],y_shape[3]])],3)

def conv2d(input_,output_dim,k_h=5,k_w=5,d_h=2,d_w=2,stddev=0.02,name="conv2d"):
	with tf.variable_scope(name):
		w=tf.get_variable("w",[k_h,k_w,input_.get_shape()[-1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv=tf.nn.conv2d(input_,w,strides=[1,d_h,d_w,1],padding="SAME")
		biases=tf.get_variable("biases",[output_dim],initializer=tf.constant_initializer(0.))
		conv=tf.reshape(conv+biases,conv.get_shape())
		return conv
def deconv2d(input_,output_shape,k_h=5,k_w=5,d_h=2,d_w=2,stddev=0.02,name="deconv2d",with_w=False):
	with tf.variable_scope(name):
		w=tf.get_variable("w",[k_h,k_w,output_shape[-1],input_.get_shape()[-1]],\
											initializer=tf.random_normal_initializer(stddev=stddev))
		deconv=tf.nn.conv2d_transpose(input_,w,output_shape=output_shape,strides=[1,d_h,d_w,1])
		biases=tf.get_variable("biases",[output_shape[-1]],initializer=tf.constant_initializer(0.))
		deconv=tf.reshape(deconv+biases,deconv.get_shape())
		if with_w:
			return deconv,w,biases
		else:
			return deconv
###Image save function
def save_images(images,size,image_path):
	img2save=inverse_transform(images)
	imsave(img2save,size,image_path)
	return img2save
def inverse_transform(images):
	return (images+1.)/2
def imsave(images,size,path):
	return scipy.misc.imsave(path,merge(images,size))
def merge(images,size):
	h,w=images.shape[1],images.shape[2]
	img=np.zeros((h*size[0],w*size[1],3))
	for idx,image in enumerate(images):
		i=idx%size[1]
		j=idx//size[1]
		img[j*h:j*h+h,i*w:i*w+w,:]=image
	return img

class DCGAN(object):
	def __init__(self,sess,image_size=108,is_crop=True,batch_size=64,sample_size=64,output_size=64,y_dim=None,z_dim=100,\
							gf_dim=64,df_dim=64,gfc_dim=1024,dfc_dim=1024,c_dim=3,dataset_name="default",checkpoint_dir=None,sample_dir=None):
		self.sess=sess
		self.is_crop=is_crop
		self.is_grayscale=(c_dim==1)
		self.batch_size=batch_size
		self.image_size=image_size
		self.sample_size=sample_size
		self.output_size=output_size
		self.y_dim=y_dim###Conditional vector
		self.z_dim=z_dim
		self.gf_dim=gf_dim		
		self.df_dim=df_dim
		self.gfc_dim=gfc_dim
		self.dfc_dim=dfc_dim
		self.c_dim=c_dim
		self.dataset_name=dataset_name
		self.checkpoint_dir=checkpoint_dir
		#Batch Normalization

		##Build model
		self.build_model()

	def generator(self,z,y=None):
		with tf.variable_scope("generator") as scope:
			s=self.output_size
			s2,s4=(int)(s/2),(int)(s/4)
			yb=tf.reshape(y,[self.batch_size,1,1,self.y_dim])

			z=tf.concat([z,y],1)
			h0=tf.nn.relu(batch_norm(linear(z,self.gfc_dim)))
			h0=tf.concat([h0,y],1)

			h1=tf.nn.relu(batch_norm(linear(h0,self.gf_dim*2*s4*s4)))
			h1=tf.reshape(h1,[self.batch_size,s4,s4,self.gf_dim*2])
			h1=conv_cond_concat(h1,yb)

			h2=tf.nn.relu(batch_norm(deconv2d(h1,[self.batch_size,s2,s2,self.gf_dim*2],name="g_h2")))
			h2=conv_cond_concat(h2,yb)
			out=tf.nn.sigmoid(deconv2d(h2,[self.batch_size,s,s,self.c_dim],name="g_h3"))
			return out

	def discriminator(self,image,y=None,reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()
			yb=tf.reshape(y,[self.batch_size,1,1,self.y_dim])
			x=conv_cond_concat(image,yb)
			
			h0=lrelu(conv2d(x,self.c_dim+self.y_dim,name="d_h0_conv"))
			h0=conv_cond_concat(h0,yb)

			h1=lrelu(batch_norm(conv2d(h0,self.df_dim+self.y_dim,name="d_h1_conv")))
			h1=tf.reshape(h1,[self.batch_size,-1])
			h1=tf.concat([h1,y],1)
			
			h2=lrelu(batch_norm(linear(h1,self.dfc_dim)))
			h2=tf.concat([h2,y],1)
			h3=linear(h2,1)
			out=tf.nn.sigmoid(h3)
			return out,h3

	def sampler(self,z,y=None):
		with tf.variable_scope("generator") as scope:
			scope.reuse_variables()
			s=self.output_size
			s2,s4=(int)(s/2),(int)(s/4)
			yb=tf.reshape(y,[self.batch_size,1,1,self.y_dim])

			z=tf.concat([z,y],1)
			h0=tf.nn.relu(batch_norm(linear(z,self.gfc_dim)))
			h0=tf.concat([h0,y],1)

			h1=tf.nn.relu(batch_norm(linear(h0,self.gf_dim*2*s4*s4),train=False))
			h1=tf.reshape(h1,[self.batch_size,s4,s4,self.gf_dim*2])
			h1=conv_cond_concat(h1,yb)

			h2=tf.nn.relu(batch_norm(deconv2d(h1,[self.batch_size,s2,s2,self.gf_dim*2],name="g_h2"),train=False))
			h2=conv_cond_concat(h2,yb)
			out=tf.nn.sigmoid(deconv2d(h2,[self.batch_size,s,s,self.c_dim],name="g_h3"))
			return out
	###Define G,D,Loss
	def build_model(self):
		self.y=tf.placeholder(tf.float32,[self.batch_size,self.y_dim],name="y")
		
		_imgsize=[self.batch_size,self.output_size,self.output_size,self.c_dim]
		self.images=tf.placeholder(tf.float32,_imgsize,name="real_images")
		
		_simgsize=[self.batch_size,self.output_size,self.output_size,self.c_dim]
		self.sample_images=tf.placeholder(tf.float32,_simgsize,name="sample_image")

		self.z=tf.placeholder(tf.float32,[None,self.z_dim],name="z")
		self.z_sum=histogram_summary("z",self.z)

		self.G=self.generator(self.z,self.y)
		self.D,self.D_logits=self.discriminator(self.images,self.y,reuse=False)
		self.D_,self.D_logits_=self.discriminator(self.G,self.y,reuse=True)
		self.sampler=self.sampler(self.z,self.y)

		self.d_sum=histogram_summary("d",self.D)
		self.d__sum=histogram_summary("d_",self.D_)
		self.g_sum=image_summary("g",self.G)

		###Loss Define
		self.d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,labels=tf.ones_like(self.D)))
		self.d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,labels=tf.zeros_like(self.D_)))
		self.g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,labels=tf.ones_like(self.D_)))
		self.d_loss=self.d_loss_real+self.d_loss_fake
		
		self.d_loss_real_sum=scalar_summary("d_loss_real",self.d_loss_real)
		self.d_loss_fake_sum=scalar_summary("d_loss_fake",self.d_loss_fake)
		self.d_loss_sum=scalar_summary("d_loss",self.d_loss)
		self.g_loss_sum=scalar_summary("g_loss",self.g_loss)
		
		t_vars=tf.trainable_variables()
		self.d_vars=[var for var in t_vars if "d_" in var.name]
		self.g_vars=[var for var in t_vars if "g_" in var.name]
		self.saver=tf.train.Saver()

	def train(self,config):
		data_x,data_y=trainimgs,trainlabels
		###optimizer
		d_optim=tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1).minimize(self.d_loss,var_list=self.d_vars)
		g_optim=tf.train.AdamOptimizer(config.learning_rate,beta1=config.beta1).minimize(self.g_loss,var_list=self.g_vars)
		self.sess.run(tf.global_variables_initializer())
		self.g_sum=merge_summary([self.z_sum,self.g_sum,self.d_loss_fake_sum,self.g_loss_sum])
		self.d_sum=merge_summary([self.z_sum,self.d_sum,self.d_loss_fake_sum,self.d_loss_real_sum])
		self.writer=SummaryWriter("./logs",self.sess.graph)
		
		sample_z=np.random.uniform(-1,1,size=(self.sample_size,self.z_dim))
		sample_images=data_x[:self.sample_size]
		sample_images=np.reshape(sample_images,[self.batch_size,self.output_size,self.output_size,1])
		sample_labels=data_y[:self.sample_size]
		print "sample labels:%s"%(np.argmax(sample_labels,axis=1))

		counter=1
		start_time=time.time()
		for epoch in xrange(config.epoch):
			batch_idxs=len(data_x)//config.batch_size
			randpermlist=np.random.permutation(len(data_x))
			for idx in xrange(0,batch_idxs):
				batch_idx=randpermlist[idx*config.batch_size:(idx+1)*config.batch_size]
				batch_images=data_x[batch_idx]
				batch_labels=data_y[batch_idx]
				batch_z=np.random.uniform(-1,1,[config.batch_size,self.z_dim]).astype(np.float32)
				
				batch_images=np.reshape(batch_images,[self.batch_size,self.output_size,self.output_size,1])
				_,summary_str=self.sess.run([d_optim,self.d_sum],feed_dict={self.images:batch_images,self.z:batch_z,self.y:batch_labels})
				#self.writer.add_summary(summary_str,counter)


				_,summary_str=self.sess.run([g_optim,self.g_sum],feed_dict={self.z:batch_z,self.y:batch_labels})
				#self.writer.add_summary(summary_str,counter)

				_,summary_str=self.sess.run([g_optim,self.g_sum],feed_dict={self.z:batch_z,self.y:batch_labels})
				#self.writer.add_summary(summary_str,counter)			

				errD_fake=self.d_loss_fake.eval({self.z:batch_z,self.y:batch_labels})
				errD_real=self.d_loss_real.eval({self.images:batch_images,self.y:batch_labels})
				errG=self.g_loss.eval({self.z:batch_z,self.y:batch_labels})
				
				counter+=1
				#print counter
				if counter%100==1:
					print "Epoch:[%2d/%3d] [%4d/%4d] time:%4.4f,d_loss:%.8f,g_loss:%.8f"%(epoch,config.epoch,idx,batch_idxs,\
								time.time()-start_time,errD_fake+errD_real,errG)
				##SAMPLE IMAGES					
				if counter%500==1:
					samples,d_loss,g_loss=self.sess.run([self.sampler,self.d_loss,self.g_loss],feed_dict={\
					self.z:sample_z,self.images:sample_images,self.y:sample_labels})
					imgname="./{}/train_{:02d}_{:04d}.png".format(config.sample_dir,epoch,idx)
					save_images(samples,[8,8],imgname)
					currsampleimg=imread(imgname)
					plt.imshow(currsampleimg)
					plt.title(imgname)
					plt.show()
					print "[Sample] d_loss:%.8f,g_loss:%.8f"%(d_loss,g_loss)

"""
Run Model
"""
if not os.path.exists(FLAGS.checkpoint_dir):
	os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
	os.makedirs(FLAGS.sample_dir)
with tf.Session() as sess:
	dcgan=DCGAN(sess,image_size=FLAGS.image_size,batch_size=FLAGS.batch_size,y_dim=10,output_size=28,\
		c_dim=1,dataset_name=FLAGS.dataset,is_crop=FLAGS.is_crop,checkpoint_dir=FLAGS.checkpoint_dir,sample_dir=FLAGS.sample_dir)
	dcgan.train(FLAGS)
	

	
				

