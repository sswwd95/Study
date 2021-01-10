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
model.fit(x_train, y_train, epochs=100, validation_split=0.2) # train_size = 0.8 에서 0.2 니까 16

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test) 
print(y_predict)

# 결과치는 주석해서 달아놓기

# shuffle = false
#loss :  0.017558246850967407
#mae :  0.13092155754566193

# Shuffle = true    -> ?  보통 트루가 값이 더 좋게 나옴. 
# loss :  0.17716486752033234
# mae :  0.3572970926761627

#validation = 0.2   -> 성능이 좋아짐, 안좋아질수도 있다. 
# loss :  0.07724357396364212
# mae :  0.22822844982147217
