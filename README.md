# tensorflow-examples
经典tensorflow的实现例子</br>
1.multilayer_perceptron:多层感知机对MNIST手写数字集进行训练识别。</br>
  网络结构：两层隐藏层，一层输出层，共三层。</br>
  激活函数：Relu</br>
  wieghts&biases初始化：高斯分布</br>
  optimizer:AdamOptimizer</br>
  测试集准确率：0.942</br>
2.convolution_net:卷积神经网络</br>
  训练时，使用dropout；预测时，不用dropout</br>
  测试集准确率：0.976562</br>
3.lstm:Long Short Term Memory</br>
  一层的lstm，其中的结点数为128</br>
  测试集准确率：0.984375</br>
