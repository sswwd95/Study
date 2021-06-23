import tensorflow as tf
tf.random.set_seed(66)

x_train = [[1.,2.],[3.,4.]]
y_train = [[1.,2.],[3.,4.]]

W = tf.Variable(tf.random_normal([2,2]), name = 'weight')
b = tf.Variable(tf.random_normal([1,2]), name = 'bias')

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
print('\ninitial weight : ', sess.run(W), '\ninitial bias : ', sess.run(b))

hypothesis = tf.matmul(x_train, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # 로스함수 / mse

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate= 0.01)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(3):
    sess.run(train)
    if step %1 == 0:
        print('\nstep : ', step, '\ncost : ', sess.run(cost), '\nweight : ', sess.run(W), '\nbias : ', sess.run(b))


'''
initial weight :  [[0.06524777 0.870543  ]
 [0.68193936 1.86472   ]]
initial bias :  [[ 1.4264158  -0.09901392]]

step :  0
cost :  8.453341 
weight :  [[0.03572131 0.76846576]
 [0.6363856  1.7202804 ]]
bias :  [[ 1.4103885  -0.14137624]]

step :  1
cost :  6.084643
weight :  [[0.01118048 0.68245035]
 [0.5979349  1.5987011 ]]
bias :  [[ 1.3964785 -0.1769402]]

step :  2
cost :  4.405374
weight :  [[-0.00916356  0.6099576 ]
 [ 0.56546444  1.4963677 ]]
bias :  [[ 1.3843521  -0.20678085]]
'''