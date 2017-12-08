import skimage
import skimage.io
import skimage.transform
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

synset = [l.strip() for l in open('model/synset.txt').readlines()]
 
def load_image(path):
  # load image
  img = skimage.io.imread(path)
  img = img/ 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = skimage.transform.resize(crop_img, (224, 224))
  return resized_img
  
def load_image_by_PIL(path,size):
	img=Image.open(path)
	img=img.resize((size,size))
	img=np.array(img)/255.
	return img
	
def print_prob(prob):
  pred = np.argsort(prob)[::-1]##argsort返回从小到大的索引值，因此使用[::-1]进行逆序
  # Get top1 label
  top1 = synset[pred[0]]
  # Get top5 label
  top5 = [synset[pred[i]] for i in range(5)]
  return top1

##加载模型  
with open("model/vgg16-20160129.tfmodel", mode='rb') as f:
  fileContent = f.read()

##定义输入
images = tf.placeholder("float", [None, 224, 224, 3])
  
###定义Graph
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
 
tf.import_graph_def(graph_def, input_map={ "images": images })
 
graph = tf.get_default_graph()

##强制使用cpu
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)
init = tf.initialize_all_variables()
sess.run(init)

result=[]
imgs=[]
for i in ['cat.jpg','airplane.jpg','zebra.jpg','pig.jpg']:
  img=load_image('model/pic/'+i)
  plt.imshow(img)
  plt.show()
  imgs.append(img)
img_num=len(imgs)
  
batch = np.array(imgs).reshape((img_num, 224, 224, 3))
assert batch.shape == (img_num, 224, 224, 3)
feed_dict = { images: batch }
prob_tensor = graph.get_tensor_by_name("import/prob:0")
prob = sess.run(prob_tensor, feed_dict=feed_dict)

for i in range (img_num): 
	result.append(print_prob(prob[i]))
 
print ("The category is :",result)
sess.close()
