from sklearn.datasets import load_diabetes # 회귀
import tensorflow as tf

x = tf.placeholder(tf.float32,shape=[None,10])
y = tf.placeholder(tf.float32, shape=[None,1])

#실습! 만들기