import tensorflow as tf
import numpy as np

tf.set_random_seed(66)

dataset = np.loadtxt('../data/csv/data-01-test-score.csv',delimiter=",")

#실습

# 아래값 predict할 것
# 73,80,75,152
# 93,88,93,185
# 89,91,90,180
# 96,98,100,196
# 73,66,70,142

print(dataset)

x_data = dataset[5:,:-1]

y_data = dataset[:,-1:]
# y_data = dataset[:, [-1]]
x_test = dataset[:5,:-1]
# x_test = [[73. ,80. ,75.],
#           [93.,88.,93.],
#           [89.,91.,90.],
#           [96.,98.,100.],
#           [73.,66.,70.]]

print(x_data.shape, y_data.shape) #(25, 3) (25, 1)

print(x_data)
# 행렬 단위로 넣어주자

x = tf.placeholder(tf.float32, shape=[None, 3]) #(25,3)

y = tf.placeholder(tf.float32, shape=[None,1]) #(25,1)


w = tf.Variable(tf.random_normal([3,1]), name='weight')
#  x의 shape=[None, 3] 에서 y의 shape=[None, 1]이 되려면 [3,1]이 되어야 한다.

b = tf.Variable(tf.random_normal([1]), name='bias')
# bias는 1개여서 1이다

# hypothesis = x * w + b
hypothesis = tf.matmul(x,w) + b
# matmul (matrix multiplication) 행렬 곱셈

# 실습

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.8)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        _, cost_val, h_val = sess.run([train, cost, hypothesis], 
                                    feed_dict={x: x_data, y: y_data}) 
        if step % 20 ==0:
            print('\n eopch : ',step, "\n Cost:", cost_val, "\n hypothesis :\n", h_val)
    predict = sess.run(hypothesis, feed_dict={x : x_test})
    print("실습 예측 결과 : "'\n',predict," \n 실제값 :\n", y_data)
'''           
eopch :  2000 Cost: 6.502113 
 hypothesis :
 [[152.83237]
 [184.37337]
 [181.15125]
 [198.89238]
 [139.52911]
 [105.56267]
 [151.17966]
 [115.24115]
 [174.5012 ]
 [165.3332 ]
 [144.21027]
 [143.15564]
 [185.48016]
 [151.9521 ]
 [152.16571]
 [188.58008]
 [142.97633]
 [182.2255 ]
 [176.46866]
 [158.14732]
 [176.75229]
 [174.5355 ]
 [168.00479]
 [150.22618]
 [190.18063]]
'''
# 실습 예측 결과 : 
#  [[152.83249]
#  [184.37338]
#  [181.15134]
#  [198.89235]
#  [139.52919]]
'''
# 아담 lr : 0.8
eopch :  5000
 Cost: 5.7378016
 hypothesis :
 [[152.6072 ]
 [185.08011]
 [181.78157]
 [199.74521]
 [139.17471]
 [103.69392]
 [150.26321]
 [112.82069]
 [174.55948]
 [164.49442]
 [143.42337]
 [142.23416]
 [186.54285]
 [152.40793]
 [151.24408]
 [189.12495]
 [143.50218]

 실습 예측 결과 :
 [[152.60727]
 [185.0802 ]
 [181.78166]
 [199.7453 ]
 [139.17477]]
'''