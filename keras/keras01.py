import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense #가장 기본적인 Dense층. 

# input_dim 이 input, Dense는 output . input_dim을 1로 설정해서 하나에서 나가고 Dense 5면 5개에서 output해서 밑에 3개에서 받는다. 그 밑에 3개에서 받았으니 input은 따로 명시 안한다.)
# input_dim 에서 dim은 dimension 차원

model = Sequential()
model.add(Dense(5, input_dim=1, activation='linear')) #가중치를 구하는 activation. linear은 relu-> 평타 85프로. 처음엔 여기까지만 이해하기.
model.add(Dense(3, activation='linear'))       # model.add(dense( )) 3, 4 -> 중간은 히든레이어
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #최소의 loss를 구하는 건 mse. adam 일단 쓰면 평타 85프로
model.fit(x, y, epochs=1000, batch_size = 1) #훈련하는 fit. 정제된 데이터 x,y를 명시해주고 몇 번 훈련하겠다는 의미의 epochs 설정. epochs=100이면 loss가 0에 수렴하기 위해 계속 작업하는데 선을 100번 그어서 훈련
# batch_size는 3개짜리 데이터를 한번에 돌리는 것 보다 한개씩 작업한다는 것. 다 집어넣고 돌리면 3. 전체 데이터보다 batch_size 더 크게 잡을 수 있다. 

#4. 평가, 예측
loss = model.evaluate(x,y, batch_size=1)   #evaluate 평가. x, y 값 넣어서  loss를 찾는다. 원래는 x, y넣으면 안됨. 답안지 보고 평가하는 것과 같음. loss는 낮아야 좋다. 선과 값과의 거리. 거리가 좁아야 좋음. 
print("loss : ", loss)

x_pred = np.array([4])
result = model.predict(x_pred)   

# result = model.predict([4])
print('result : ', result)

