from sklearn.datasets import load_boston # 회귀
import tensorflow as tf
from sklearn.metrics import r2_score

# 실습 : 최종 sklearn의 accuracy_score값으로 결론낼 것

dataset = load_boston()

x_data = dataset.data
y_data = dataset.target

print(x_data.shape, y_data.shape)
# (506, 13) (506,)
y_data = y_data.reshape(-1,1)
print(x_data.shape, y_data.shape)
# (506, 13) (506, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size = 0.8, random_state = 66)


x = tf.placeholder(tf.float32,shape=[None,13])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([13,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x,w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

train = tf.train.AdamOptimizer(learning_rate=0.8).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        _,cost_val, h_val = sess.run([train, cost, hypothesis],feed_dict={x:x_train, y:y_train})
        
        y_pred = sess.run(hypothesis, feed_dict={x:x_test})
        
        if step % 1000 ==0 :
            print(step, cost_val, h_val)
    
    print("R2 : ", r2_score(y_test,y_pred))


#R2 :  0.8308446570473413





