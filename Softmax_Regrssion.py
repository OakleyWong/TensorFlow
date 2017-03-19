# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 23:24:14 2017

@author: Yunfan Wang
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)#one_hot编码，1=[10000 00000]
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)
import tensorflow as tf
sess = tf.InteractiveSession()#开辟一个区域
#----------------定义算法forward公式--------------------------
x=tf.placeholder(tf.float32,[None,784])#placeholder 输入数据的地方
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)#y为预测Label
y_=tf.placeholder(tf.float32,[None,10])#y_为存放输入真实label
#----------------定义loss选定优化器--------------------------
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#---------------训练----------------------------------------
tf.global_variables_initializer().run()
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})
#--------------验证------------------------------------------
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))