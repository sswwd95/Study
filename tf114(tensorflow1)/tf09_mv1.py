import tensorflow as tf
tf.set_random_seed(66)

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]


x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
# placeholder를 통해 위의 값들을 넣어주고

w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
# 변수초기화

hypothesis = x1*w1 + x2*w2 + x3*w3 +b


cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost) # cost를 최소화시키도록 훈련

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, h_val, _ = sess.run([cost, hypothesis, train], 
                        feed_dict={x1:x1_data, x2:x2_data, x3 : x3_data, y : y_data})
    if step % 10 ==0:
        print(step, "cost : ",cost_val, "\n", "hypothesis : ",h_val)
sess.close()