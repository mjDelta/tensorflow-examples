# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:41:29 2017

@author: ZMJ
"""
import scipy.io
import numpy as np
import os
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf

print("Package Loaded!")

"""
Ops Define
"""
def _conv_layer(input,weights,bias):
  conv=tf.nn.conv2d(input,tf.constant(weights),strides=(1,1,1,1),padding="SAME")
  return conv+bias

def _pool_layer(input):
  return tf.nn.max_pool(input,ksize=(1,2,2,1),strides=(1,2,2,1),padding="SAME")

def preprocess(image,mean_pixel):
  return image-mean_pixel

def unprocess(image,mean_pixel):
  return image+mean_pixel

def imread(path):
  return scipy.misc.imread(path).astype(np.float)

def imsave(path,img):
  img=np.clip(img,0,255).astype(np.int8)
  scipy.misc.imsave(path,img)

"""
Netword Define
"""
def net(data_path,input_image):
  layers={
      "conv1_1","relu1_1","conv1_2","relu1_2","pool1",
      "conv2_1","relu2_1","conv2_2","relu2_2","pool2",
      "conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","conv3_4","relu3_4","pool3",
      "conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","conv4_4","relu4_4","pool4",
      "conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","conv5_4","relu5_4"
      }
  data=scipy.io.loadmat(data_path)
  mean=data["normalization"][0][0][0]
  mean_pixel=np.mean(mean,axis=(0,1))
  weights=data["layers"][0]
  net={}
  current=input_image
  for i,name in enumerate(layers):
    kind=name[:4]
    if kind=="conv":
      k,b=weights[i][0][0][0][0]
      ##mat:[width,height,in_channel,out_channel]
      ##tf:[height,width,in_channel,out_channel]
      k=k.transpose(k,(1,0,2,3))
      b=b.reshape(-1)
      current=_conv_layer(current,k,b)
    elif kind=="relu":
      current=tf.nn.relu(current)
    else:
      current=_pool_layer(current)
    net[name]=current
  assert len(net)==len(layers)
  return net,mean_pixel,layers

"""
Use Params and Model
"""
cwd=os.getcwd()
VGG_PATH=cwd+"/data/imagenet-vgg-verydeep-19.mat"
IMG_PATH=cwd+"/images/cat.jpg"
input_image=imread(IMG_PATH)#[h,w,ch]
shape=(1,)+input_image.shape#[1,h,w,ch]
with tf.Graph().as_default(),tf.Session() as sess:
  image=tf.placeholder("float",shape=shape)
  nets,mean_pixel,all_layers=net(VGG_PATH,image)
  input_image_pre=np.array([preprocess(input_image,mean_pixel)])
  layers=all_layers
  for i,layer in enumerate(layers):
    print("[%d/%d]%s"%(i+1,len(layers),layer))
    features=nets[layer].eval(feed_dict={image:input_image_pre})
    print("Type of features is %s"%type(features))
    print("Shape of features is %s"%(features.shape))
    
    """
    Visualization Every Layer's First Channel
    """
    plt.figure(i+1,figsize=(10,5))
    plt.matshow(features[0,:,:,0],cmap=plt.cm.gray,fignum=i+1)
    plt.title(""+layer)
    plt.colorbar()
    plt.show()
