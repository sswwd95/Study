import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype=np.float32)

# 다층 레이어를 연결해보자, 레이어마다 w가 있다 (딥러닝구성)
#------------------------------------------------------------
x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

w1 = tf.Variable(tf.random_normal([2,10]),name='weight1') # 히든 레이어 노드는 임의로 설정, 만약 10개라면?
b1 = tf.Variable(tf.random_normal([10]),name='bias1')
layer1 = tf.sigmoid(tf.matmul(x,w1) + b1)
# model.add(Dense(10, input_dim=2, activation='sigmoid'))

w2 = tf.Variable(tf.random_normal([10,7]),name = 'weight2')
b2 = tf.Variable(tf.random_normal([7]),name = 'bias2')
layer2 = tf.sigmoid(tf.matmul(layer1,w2) + b2) # layer1에서 나온 새로운 x값과 w를 곱해준다
# model.add(Dense(7, activation='sigmoid'))

w3 = tf.Variable(tf.random_normal([7,1]),name = 'weight3')
b3 = tf.Variable(tf.random_normal([1]),name = 'bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2,w3) + b3) # output, y와 동일해야 한다
# model.add(Dense(1, activation='sigmoid'))


cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
# binary crossentropy 식

train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

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
    
# 5000 3.4719774e-06
# 예측값 :
#  [[1.7386005e-06]
#  [9.9999642e-01]
#  [9.9999642e-01]
#  [5.0252097e-06]]
#  원래값 :
#  [[0.]
#  [1.]
#  [1.]
#  [0.]]
#  acc : 1.0

# tf14_xor1 파일에서 왜 0.75만 나왔을까? deep러닝이 아니기 때문! 
# 다층레이어로 연결해주고 optimizer를 adam으로 바꿔주면 acc 1.0이 나온다

# 노드를 64,32 로 바꿔줘도 1.0 나온다
'''
import tensorflow as tf
import numpy as np
tf.set_random_seed(66)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]],dtype=np.float32)

# 다층 레이어를 연결해보자, 레이어마다 w가 있다 (딥러닝구성)
#------------------------------------------------------------
x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

w1 = tf.Variable(tf.random_normal([2,64]),name='weight1') # 히든 레이어 노드는 임의로 설정, 만약 10개라면?
b1 = tf.Variable(tf.random_normal([64]),name='bias1')
layer1 = tf.sigmoid(tf.matmul(x,w1) + b1)
# model.add(Dense(10, input_dim=2, activation='sigmoid'))

w2 = tf.Variable(tf.random_normal([64,32]),name = 'weight2')
b2 = tf.Variable(tf.random_normal([32]),name = 'bias2')
layer2 = tf.sigmoid(tf.matmul(layer1,w2) + b2) # layer1에서 나온 새로운 x값과 w를 곱해준다
# model.add(Dense(7, activation='sigmoid'))

w3 = tf.Variable(tf.random_normal([32,1]),name = 'weight3')
b3 = tf.Variable(tf.random_normal([1]),name = 'bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2,w3) + b3) # output, y와 동일해야 한다
# model.add(Dense(1, activation='sigmoid'))

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



# 5000 0.11174306
# 예측값 :
#  [[0.08428652]
#  [0.8810192 ]
#  [0.9068833 ]
#  [0.12573059]]
#  원래값 :
#  [[0.]
#  [1.]
#  [1.]
#  [0.]]
#  acc : 1.0
'''