from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop 
# RMSprop? 과거의 모든 기울기를 균일하게 더하지 않고 새로운 기울기의 정보만 반영하도록 해서 학습률이 크게 떨어져 0에 가까워지는 것을 방지하는 방법이다.
import numpy as np
import matplotlib.pyplot as plt

# 1 데이터
x = np.arange(1,11)
y = np.array([1,2,4,3,5,5,7,9,8,11])
print('\n',x,'\n',y)

# 2. 모델구성
model = Sequential()
model.add(Dense(1,input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
optimizer = RMSprop(learning_rate=0.01)

model.compile(loss='mse', optimizer=optimizer)
model.fit(x,y, epochs=10)

y_pred = model.predict(x)

plt.scatter(x,y) #scatter = 흩뜨림. 데이터를 하나하나 점 찍겠다는 것
plt.plot(x, y_pred, color='red')
plt.show()

# 딥러닝에서는 히든레이어가 있으면 성능 높아짐. 에폭이 커지면 성능 높아짐.

# 하지만 머신러닝에서는 히든레이어가 없다. 히든레이어가 없으면 연산량이 적다는 것. = 속도 빠름

# 01.27 머신러닝 시작!