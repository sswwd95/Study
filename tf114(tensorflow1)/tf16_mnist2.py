# 선생님 코드
import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score


# 1. 데이터
(x_train, y_train),(x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float',[None,10])

w = tf.Variable(tf.random_normal([784,10]), name = 'weight')
b = tf.Variable(tf.random_normal([1,10]), name = 'bias')

# 2. 모델구성

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b) # 결과 값을 감싸서 다음 레이어에 보내준다

loss = tf.reduce_mean(-tf.reduce_mean(y*tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 변수 초기화

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_train, y:y_train})
        if step % 200 ==0:
            print(step, cost_val)
