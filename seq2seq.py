# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 21:24:56 2017

@author: ZMJ
"""
import numpy as np
import tensorflow as tf
import collections
import argparse
import time
import os
from six.moves import cPickle
print("Packages Imported")

data_dir="data/"
save_dir="data/"
input_file=os.path.join(data_dir,"linux.txt")
with open(input_file,"r") as f:
    data=f.read()
print("Text loaded from %s"%input_file)

counter=collections.Counter(data)
count_pairs=sorted(counter.items(),key=lambda x:-x[1])
print("Type of 'counter.items()' is %s and length is %d"%(type(counter.items),len(counter.items())))
for i in range(5):
    print("[%d/%d]"%(i,5)),
    print(list(counter.items())[i])
    
print("Type of 'count_pairs' is %s and length is %d"%(type(count_pairs),len(count_pairs)))
for i in range(5):
    print("[%d/%d]"%(i,5)),
    print(list(count_pairs[i]))

chars,counts=zip(*count_pairs)
vocab=dict(zip(chars,range(len(chars))))
print("Type of 'chars' is %s and length is %d"%(type(chars),len(chars)))
for i in range(5):
    print("[%d/%d]"%(i,5)),
    print("vocab[%s] is %s"%(chars[i],vocab[chars[i]]))
    
with open(os.path.join(save_dir,"chars_vocab.pkl"),"wb") as f:
    cPickle.dump((chars,vocab),f)

corpus=np.array(list(map(vocab.get,data)))
print("Type of 'corpus' is %s,shape is %s,and length is %d"%(type(corpus),corpus.shape,len(corpus)))

check_len=10
print("\n 'corpus' looks like %s"%corpus[:check_len])
for i in range(check_len):
    _wordidx=corpus[i]
    print("[%d/%d] chars[%02d] corresponds to '%s'"%(i,check_len,_wordidx,chars[_wordidx]))


batch_size=50
seq_length=200
num_batches=int(corpus.size/batch_size*seq_length)





