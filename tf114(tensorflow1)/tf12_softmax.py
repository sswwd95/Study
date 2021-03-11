import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],  #2
          [0,0,1],
          [0,0,1],
          [0,1,0],  #1
          [0,1,0],
          [0,1,0],
          [1,0,0],  #0
          [1,0,0]]

x = tf.placeholder('float', [None,4])
y = tf.placeholder('float', [None,3])

w = tf.Variable(tf.random_normal([4,3]),name='weight')
b = tf.Variable(tf.random_normal([1,3]), name = 'bias')
#y의 shape에 맞춘다

hypothesis = tf.nn.softmax(tf.matmul(x,w) + b) # 결과 값을 감싸서 다음 레이어에 보내준다
# (n,4)*(4,3)=(n,3) + (1,3) = (n,3)

# cost = tf.reduce_mean(tf.square(hypothesis - y))
# 기존의 mse

# 다중분류는 categorical crossentropy
loss = tf.reduce_mean(-tf.reduce_mean(y*tf.log(hypothesis), axis=1))

# loss를 최소화시키기 위해 optimizer쓰기
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 변수 초기화

    for step in range(2001):
        _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
        if step % 200 ==0:
            print(step, cost_val)
    
    # predict
    a = sess.run(hypothesis, feed_dict={x:[[1,11,7,9]]})
    print(a,sess.run(tf.argmax(a,1)))
        
# 1600 0.23607637
# 1800 0.2275399
# 2000 0.22019444
# [[0.7818437  0.17601936 0.04213692]] [0]