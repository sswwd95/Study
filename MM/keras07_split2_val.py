
from tensorflow.keras.models import Sequential       
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(1,101))
y = np.array(range(1,101))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, train_size=0.8, shuffle=False) # shuffle=True하면 순서대로, False하면 무작위로 섞는다. False의 값이 더 좋다. 

# validation은 train 안에서 나누는 것.
x_train, x_val, y_train, y_val= train_test_split(x_train, y_train,
                                                 test_size=0.2, shuffle = False)  

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val)) # split을 해서 validation 구했으면 validation_data. train_test_split 안했으면 validation_split 쓰기.

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

#validation_split = 0.2   -> 성능이 좋아짐, 안좋아질수도 있다. 
# loss :  0.07724357396364212
# mae :  0.22822844982147217

#validation_data 
# loss :  0.022586554288864136
# mae :  0.1324489414691925
# RMSE :  0.16741932528015258
# mse :  0.028029230477261535
# R2 :  0.9999588132505404

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


