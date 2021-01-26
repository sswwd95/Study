
# custom loss


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow.keras.backend as K
'''
def costom_mse(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))
    # 원래 값에서 예측값을 뻰거를 제곱해서 그걸 평균내줬다는 뜻 = mse
    # 인자는 원래값(y_true)과 예측값(y_pred)
    # 순서대로 원래값, 예측값으로 계산함. 
    # 이름 상관없고 순서만 맞으면 된다. 

def quatile_loss(y_true, y_pred):
    qs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    q = tf.constant(np.array([qs]), dtype=tf.float32) # constant의 상수를 텐서플로우 형태의 상수형으로 들어감. 텐서플로우에 1+1 넣으면 2로 나오는게 아니라 텐서플로우 형태로 나옴.
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e) # 최댓값 (식은 알아서 찾기)
    return K.mean(v) # 평균(loss가 1개만 출력)
'''

def quantile_loss_dacon(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)
    
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8]).astype('float32') #실수형으로 바꾸는 것
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')

print(x.shape)

# 2. 모델
model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))


# 3. 컴파일, 훈련
# model.compile(loss=lambda y_true, y_pred: quantile_loss_dacon(quantiles, y_true, y_pred),optimizer='adam')
# y_true, y_pred -> 인수. lambda를 사용하면 인수를 : 뒤에 넣어준다는 것 / def함수 뒤에 quentiles를 인수로 같이 넣는다는 것.

model.compile(loss=lambda y_true, y_pred: quantile_loss_dacon(quantiles[0], y_true, y_pred),optimizer='adam')
# (quantiles[0], y_true, y_pred) 0을 넣으면 0번째인 0.1을 출력한다는 것. 

model.fit(x,y, batch_size=1, epochs=50)

# 4. 평가, 예측
loss = model.evaluate(x,y)
print(loss)

a = model.predict(x)
print(a)
# 1번 :  quatile_loss 전
# 0.0014978328254073858

# 2번 : quatile_loss 후 (최댓값에 대한 평균값 나온 것)
# 0.00820956751704216
# 지표가 달라져서 좋다 나쁘다 평가하기 어려움

# 3번 : quatile_loss[0] -> 0.1의 loss값 나온 것
# 0.01681545190513134