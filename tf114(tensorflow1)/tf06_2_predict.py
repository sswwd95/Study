

import tensorflow as tf
tf.set_random_seed(66) # 랜덤 값 고정해야 동일한 값이 출력된다

# x_train = [1,2,3]
# y_train =  [3,5,7]
x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x_train * W + b # -> y = wx + b
# hypothesis : 가설

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) 
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 자동으로 session닫아주는 것 = with문으로 감싸준다
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        # sess.run(train)
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], # 각 결과값에 대한 반환
                                    feed_dict={x_train:[1,2,3], y_train:[3,5,7]}) # train을 실행시켜라
        if step % 20 ==0: # step을 20으로 나눈 나머지가 0이면 출력한다(2000 epoch돌면서 20번마다 출력)
            # print(step, sess.run(cost), sess.run(W), sess.run(b)) 
            print(step, cost_val, W_val, b_val)# 결과값 반환
#-----------------------------------------------------------------------------------------------------
    predict = sess.run(hypothesis, feed_dict={x_train:[6,7,8]})
    print(predict)
#-----------------------------------------------------------------------------------------------------

# 실습 predict 구하기
# 1. [4]     -> [8.998177]
# 2. [5,6]   -> [10.997122 12.996066]
# 3. [6,7,8] -> [12.996066 14.99501  16.993954]


    # 동재 코드
    print('[4] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[4]}))
    print('[5, 6] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[5,6]}))
    print('[6, 7, 8] 예측결과 : ', sess.run(hypothesis, feed_dict = {x_train:[6,7,8]}))
    # [4] 예측결과 :  [8.998177]
    # [5, 6] 예측결과 :  [10.997122 12.996066]
    # [6, 7, 8] 예측결과 :  [12.996066 14.99501  16.993954]




