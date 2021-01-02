# 1 : 다 mlp 함수형
# keras10_mlp6.py를 함수형으로 바꾸시오.


import numpy as np
# 1. 데이터
x = np.array(range(100))
y = np.array([range(711,811),range(1,101), range(201,301)])
x_pred2 = np.array([101])

print(x.shape) # (3,100)
print(y.shape)   

x = np.transpose(x) 
y = np.transpose(y)   

print(x.shape)    
print(y.shape)   



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(
    x,y, random_state=66, train_size=0.8, shuffle=True)

# 랜덤 스테이트는 그냥 아무 숫자나 정하는 것.어차피 값에 따라 결과 나옴.

print(x_train.shape)   #(80,3)
print(y_train.shape)    #(80,3)

# 2. 모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input

input1=Input(shape=(1,))
dense1 = Dense(1, activation='relu')(input1)
dense2 = Dense(5)(dense1)
dense3 = Dense(5)(dense2)
outputs = Dense(3)(dense3)
model = Model(inputs = input1, outputs = outputs)
model.summary()

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