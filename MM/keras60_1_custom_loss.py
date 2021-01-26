
# custom loss


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8]).astype('float32') #실수형으로 바꾸는 것
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')

print(x.shape)

# 2. 모델
model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))

def costom_mse(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))
    # 원래 값에서 예측값을 뻰거를 제곱해서 그걸 평균내줬다는 뜻 = mse
    # 인자는 원래값(y_true)과 예측값(y_pred)
    # 순서대로 원래값, 예측값으로 계산함. 
    # 이름 상관없고 순서만 맞으면 된다. 

# 3. 컴파일, 훈련
model.compile(loss=costom_mse,optimizer='adam')

model.fit(x,y, batch_size=1, epochs=50)

# 4. 평가, 예측
loss = model.evaluate(x,y)
print(loss) #0.0014978328254073858