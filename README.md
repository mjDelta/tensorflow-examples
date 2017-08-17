# tensorflow-examples
Classical Tensorflow Examples on DeepLearning
---------------------------------------------
Mainly From:https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/ </br>
Mainly From:https://github.com/sjchoi86/Tensorflow-101</br>
But tuning some parameters to make the net perform better,and conclude some useful tricks</br>

1.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/multilayer_perceptron.py">multilayer_perceptron</a>:train dataset:MNIST</br>
>Initialisation(wieghts&biases )：`Gaussian Distribution(0,0.05)`</br>
  Test Dataset Accuracy:0.942</br>
  `Tuning Ticks`:Weights Initialisation is very important.When stddev is 0.05,its accuracy gets 0.97!!!Guess it is close to `Xavier` Initialisation(`2/(Nin+Nout`)</br>


2.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/convolution_net.py">convolution_net</a>:</br>
  >`Ticks`:When training,use dropout.When predicting,don't use dropout.</br>
  Test Dataset Accuracy:0.976562</br>

  
3.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/lstm.py">lstm</a>:Long Short Term Memory</br>
  >One layer lstm with 128 lstm units</br>
  Test Dataset Accuracy:0.984375</br>

  
4.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/bilstm.py">bilstm</a>:Bidirectional LSTM</br>
  >One hidden layer includes `forward LSTM`(128 lstm units) and `backward LSTM`(128 lstm units) meanwhile,so it has 258 lstm units.</br>
  Test Dataset Accuracy:1.0！！</br>

5.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/dynamic_lstm.py">dynamic_lstm</a>:</br>
>`Ticks`:When the Input Data's time step is different,use dynamic_lstm.</br>
Test Dataset Accuracy:0.788</br>

6.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/autoencoder.py">AutoEncoder</a>:</br>
>`Unsupervised Learning`:Both Input and Output is pictures(`Non-Label`).</br>
Visualisation:left is after decoder's picture,right is the original picture</br>
![image](https://github.com/mjDelta/tensorflow-examples/blob/master/imgs/figure_1.PNG)</br>

7.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/save_restore_model.py">Save and Restore</a>:</br>
>Saved File's suffixis `.ckpt`</br>
Saver Defination:`tf.train.Saver()`</br>
Save op:`saver.save(sess,path)`</br>
Restore op:`saver.restore(sess,path)`</br>

8.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/tensorboard_advance.py">tensorboard</a>:</br>
>Command line:`tensorboard --logdir=/tmp/tensorflow_logs`</br>
![image](https://github.com/mjDelta/tensorflow-examples/blob/master/imgs/tensorboard.PNG)</br>

9.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/linear_regression.py">linear regression</a>:</br>
>Target funcation:y=wx+b</br>
cost:Here we define it as `MSE`.We can expand it to `MAE`.etc,to do `Robust Regression`.</br>

10.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/logistic_regression.py">logistic regression<a/>：</br>
>Add activation like sigmoid and softmax on the base of linear regression.</br>
`Sigmoid` is used in binary classes</br>
`Softmax` is used in multi	classes</br>

11.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/seq2seq.py">seq2seq</a>:Dataset is linux code</br>
>1.Make chars/index dictionary

12.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/dae.py">dae</a>:denoising auto encoder</br>
>input:`corrupted` pictures(add noisy data)</br>
label:`original` pictures</br>

13.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/cae.py">cae</a>:denoising auto encoder with conv and deconv</br>
>`enconder`:use `convolution`</br>
`decoder`:use `transpose convolution`(deconvolution)</br>
![image](https://github.com/mjDelta/tensorflow-examples/blob/master/imgs/cae5.png)</br>

14.<a href="https://github.com/mjDelta/tensorflow-examples/blob/master/dcgan.py">dcgan</a>:deep convolutional GAN</br>
>`generator`:use `transpose convolution`(deconvolution)</br>
`discriminator`:use `convolution`</br>

Mainly From:https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/ </br>
Mainly From:https://github.com/sjchoi86/Tensorflow-101</br>
But tuning some parameters to make the net perform better,and conclude some useful tricks</br>
