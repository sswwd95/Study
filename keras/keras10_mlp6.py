# 1 : 다 mlp


# 실습 : 코드를 완성할 것
# mlp4처럼 predict값을 완성할 것


import numpy as np
# 1. 데이터
x = np.array(range(100))
y = np.array([range(711,811),range(1,101), range(201,301)])
print(x.shape) 
print(y.shape)   

y_pred2 = np.array([812,102,302])
print("y_pred2.shape : ", y_pred2.shape)


x = np.transpose(x) 
y = np.transpose(y)   

y_pred2 = y_pred2.reshape(1, 3)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(
    x,y, random_state=66, train_size=0.8, shuffle=True)

# 랜덤 스테이트는 그냥 아무 숫자나 정하는 것.어차피 값에 따라 결과 나옴.

print(x_train.shape)   #(80,3)
print(y_train.shape)    #(80,3)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # 텐서플로우에서 케라스 부르는게 속도 더 빠름
# from keras.layers import Dense -> 원래는 이렇게 썼는데 텐서플로우가 케라스 먹음. 이건 좀 느림

model = Sequential()
model.add(Dense(10, input_dim=1))  # 컬럼=피처=특성=열
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)



# 4. 평가 , 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mae : ', mae)

y_predict = model.predict(x_test)


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)