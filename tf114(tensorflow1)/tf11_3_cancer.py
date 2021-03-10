from sklearn.datasets import load_breast_cancer # 이진분류
import tensorflow as tf

x = tf.placeholder(tf.float32,shape=[None,30])
y = tf.placeholder(tf.float32, shape=[None,1])

#실습! 만들기