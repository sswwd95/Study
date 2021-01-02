from tensorflow.keras.models import Sequential    
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(1,101))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True)  

print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(x_test.shape)

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)  
print(y_predict)
