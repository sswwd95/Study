# 다 : 1 mlp
import numpy as np

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [11,12,13,14,15,16,17,18,19,20]]) # (2, 10) -> 2행 10열
y = np.array([1,2,3,4,5,6,7,8,9,10])
print(x.shape)   
                  
x = np.transpose(x)    
print(x)
print(x.shape) 

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x,y, epochs=100, batch_size=1, validation_split=0.2)

# 4. 평가 , 예측
loss, mae = model.evaluate(x, y)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x)