# tensorflow-examples
练习来源：https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/ </br>
经典tensorflow的实现例子</br>
1.multilayer_perceptron:多层感知机对MNIST手写数字集进行训练识别。</br>
>网络结构：两层隐藏层，一层输出层，共三层。</br>
  激活函数：Relu</br>
  wieghts&biases初始化：高斯分布</br>
  optimizer:AdamOptimizer</br>
  测试集准确率：0.942</br>
  
2.convolution_net:卷积神经网络</br>
  >训练时，使用dropout；预测时，不用dropout</br>
  测试集准确率：0.976562</br>
  
3.lstm:Long Short Term Memory</br>
  >一层的lstm，其中的结点数为128</br>
  测试集准确率：0.984375</br>
  
4.bilstm:Bidirectional LSTM</br>
  >一层隐藏层里同时包含forward LSTM（结点数128）和backward LSTM（结点数128），所以隐藏层的总结点数256.</br>
  测试集准确率：1.0！！</br>

5.dynamic_lstm:</br>
>输入数据长度不一致，动态获得LSTM Cell中的outputs.</br>
测试机准确率：0.788</br>

6.AutoEncoder:自编码器</br>
>无监督学习：encoder类似图片压缩，decoder类似图片解压;输入输出都是X(图片)</br>
示例图片：左边为decoder后的，右边为原始图片</br>
![image](https://github.com/mjDelta/tensorflow-examples/blob/master/imgs/figure_1.PNG)</br>
AutoEncoder功能：降维，降噪</br>

7.Save and Restore:保存和恢复模型</br>
>保存文件后缀：.ckpt</br>
saver定义：tf.train.Saver()</br>
保存操作:saver.save(sess,path)</br>
恢复操作：saver.restore(sess,path)</br>

8.tensorboard base:</br>
>tensorboard可视化summary信息</br>
tensorboard可视化网络详细结构
