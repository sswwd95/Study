import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype=np.float32)

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([2,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w) + b)

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
# binary crossentropy 식

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# tf.cast : boolen 형태인 경우 true이면 1, false이면 0을 출력
predict = tf.cast(hypothesis>0.5,dtype=tf.float32) # 0.5초과하면 1 
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))
# tf.equal(predict, y) -> equal이 들어가면 predict와 y의 값이 같다면 1로 출력, 아니면 0을 출력

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val,_ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})

        if step % 200 ==0:
            print(step, cost_val)
    h, p, a = sess.run([hypothesis, predict, accuracy], feed_dict={x:x_data, y:y_data})
    print('예측값 :\n', h,'\n 원래값 :\n', p, '\n acc :',a) 

# 예측값 :
#  [[0.5098468 ]
#  [0.50287294]
#  [0.5002181 ]
#  [0.4932434 ]]
#  원래값 :
#  [[1.]
#  [1.]
#  [1.]
#  [0.]]
#  acc : 0.75

