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

# 2. 모델구성
x = tf.placeholder('float',[None,784])
y = tf.placeholder('float',[None,10])

# ------------------ 첫번째 레이어---------------------
# w1 = tf.Variable(tf.random_normal([784,100]), name = 'weight1')
# get_variable과 Variable의 차이점? 메모리 구성에 차이가 있긴 하지만 거의 비슷, 변수를 구한다
w1 = tf.get_variable('weight1', shape=[784,100],
                    initializer=tf.contrib.layers.xavier_initializer()) 
                    # xavier_initializer = kernel_initializer
print('w1 : ', w1)
# w1 :  <tf.Variable 'weight1:0' shape=(784, 100) dtype=float32_ref>

b1 = tf.Variable(tf.random_normal([100]), name = 'bias1')
print('b1:',b1)
# b1: <tf.Variable 'bias1:0' shape=(100,) dtype=float32_ref>

# layer1 = tf.nn.softmax(tf.matmul(x,w1) + b1) # 히든레이어에 softmax있으면 잘 안돌아간다
# layer1 = tf.nn.elu(tf.matmul(x,w1) + b1) 
# layer1 = tf.nn.selu(tf.matmul(x,w1) + b1) 
layer1 = tf.nn.relu(tf.matmul(x,w1) + b1)
print('layer1 : ', layer1)
# layer1 :  Tensor("Elu:0", shape=(?, 100), dtype=float32)

layer1 = tf.nn.dropout(layer1, keep_prob=0.3) 
# keep_prob = 0.3 :30% drop out한다는 것

print('layer1 : ', layer1)
# layer1 :  Tensor("dropout/mul_1:0", shape=(?, 100), dtype=float32)
# ----------------------------------------------------------------------

w2 = tf.get_variable('weight2', shape=[100,128],
                    initializer=tf.contrib.layers.xavier_initializer()) 
b2 = tf.Variable(tf.random_normal([128]), name = 'bias2')
layer2 = tf.nn.relu(tf.matmul(layer1,w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.3) 

w3 = tf.get_variable('weight3', shape=[128,64],
                    initializer=tf.contrib.layers.xavier_initializer()) 
b3 = tf.Variable(tf.random_normal([64]), name = 'bias3')
layer3 = tf.nn.relu(tf.matmul(layer2,w3) + b3)
layer3 = tf.nn.dropout(layer3, keep_prob=0.3) 

w4 = tf.get_variable('weight4', shape=[64,10],
                    initializer=tf.contrib.layers.xavier_initializer()) 
b4 = tf.Variable(tf.random_normal([10]), name = 'bias4')
hypothesis = tf.nn.softmax(tf.matmul(layer3,w4) + b4)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_mean(y*tf.log(hypothesis), axis=1))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

# training_epochs = 15
# batch_size = 100
training_epochs = 100
batch_size = 100
total_batch = int(len(x_train)/batch_size)
# 60000/100 = 600, 1에폭당 600번

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_loss= 0

    for i in range(total_batch):  # 600번 돈다
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        # 1에폭에서 첫 번째 배치는 100개의 데이터. batch_x, y ->  0~100번째까지
        feed_dict = {x:batch_x, y:batch_y}
        l, _ = sess.run([loss,optimizer], feed_dict=feed_dict)
        # 100개의 cost라서 나머지 59900개도 해야한다
        avg_loss += l/total_batch
        # avg_cost? cost구한거에서 600을 나누기때문에 전체 cost가 나온다
        # 1에폭에 600번이 도는데 한 번 돌때마다 loss가 나온다 
        # 600개의 loss를 다 더해서 다시 600으로 나누면 loss의 평균값이 나온다 = 6만개의 loss 평균값이라고 생각할 수 있다
        # 데이터가 너무 크니까 한번에 6만개 돌면 터진다 그래서 나눠서 600개만 한다
    print('epoch : ', '%04d' %(epoch+1),
          'loss = {:.9f}'.format(avg_loss))
print("끝")

prediction = tf.equal(tf.math.argmax(hypothesis, 1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('acc :', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))