# 다 : 1 mlp
import numpy as np

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [11,12,13,14,15,16,17,18,19,20]]) # (2, 10) -> 2행 10열
y = np.array([1,2,3,4,5,6,7,8,9,10]) # -> output =  1

print(x.shape)  #(2, 10)
print(y.shape)  #(10,)
                  
x = np.transpose(x) # 행, 열의 위치를 바꿔준다. 모델에 입력하려면 행과 열이 맞아야한다.    
print(x)
# [[ 1 11]
#  [ 2 12]
#  [ 3 13]
#  [ 4 14]
#  [ 5 15]
#  [ 6 16]
#  [ 7 17]
#  [ 8 18]
#  [ 9 19]
#  [10 20]]
print(x.shape)  #(10, 2)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

model = Sequential()
model.add(Dense(10, input_dim=2)) # 컬럼이 2개다. 컬럼=피처=특성=열
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1)) # -> output =  1

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x,y, epochs=100, batch_size=1, validation_split=0.2)

# 4. 평가 , 예측
loss, mae = model.evaluate(x, y)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x)