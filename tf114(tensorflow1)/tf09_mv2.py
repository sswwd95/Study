import tensorflow as tf
tf.set_random_seed(66)


x_data = [[73, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]

y_data = [[152],
          [185],
          [180],
          [205],
          [142]]


# x_data = [[73, 80, 75],[93, 88, 93], 
#           [85, 91, 90],[96, 98, 100],[73, 66, 70]]
# y_data = [[152],[185],[180],[196],[142]]

# 행렬 단위로 넣어주자

x = tf.placeholder(tf.float32, shape=[None, 3]) #(5,3)

y = tf.placeholder(tf.float32, shape=[None, 1]) #(5,1)

w = tf.Variable(tf.random_normal([3,1]), name='weight')
#  x의 shape=[None, 3] 에서 y의 shape=[None, 1]이 되려면 [3,1]이 되어야 한다.

b = tf.Variable(tf.random_normal([1]), name='bias')
# bias는 1개여서 1이다

# hypothesis = x * w + b
hypothesis = tf.matmul(x,w) + b
# matmul (matrix multiplication) 행렬 곱셈

# 실습

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, h_val = sess.run([train, cost, hypothesis], 
                                    feed_dict={x: x_data, y: y_data}) 
        if step % 20 ==0:
            print('eopch : ',step, "Cost:", cost_val, "\n hypothesis :\n", h_val)

'''           
eopch :  1980 Cost: 296.03598
 hypothesis :
 [[171.27554]
 [191.23407]
 [151.92091]
 [212.78763]
 [127.14444]]
eopch :  2000 Cost: 296.03317 
 hypothesis :
 [[171.27557]
 [191.23401]
 [151.92111]
 [212.78754]
 [127.14451]]
'''
