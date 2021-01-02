from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
# 1. 데이터
x = np.array([range(100), range(201, 301), range(401,501),
              range(601,701),range(801,901)])
y = np.array([range(811,911),range(1,101)])

print(x.shape)  # (5,100)
print(y.shape)  # (2,100)
x_pred2 = np.array([100,302,502,702,1001])
print("x_pred2.shape : ", x_pred2.shape)

x = np.transpose(x) 
y = np.transpose(y)      
# x_pred2 = np.transpose(x_pred2)
x_pred2 = x_pred2.reshape(1, 5)

print(x.shape)    #(100,5)
print(y.shape)    #(100,2)
print(x_pred2.shape)
print("x_pred2.shape : ", x_pred2.shape)  #(1,5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(
    x,y, random_state=66, train_size=0.8, shuffle=True)

print(x_train.shape)   #(80,5)
print(y_train.shape)    #(80,2)

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=5))  # 컬럼=피처=특성=열
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
         validation_split=0.2, verbose=2)

'''
verbose==없이 :
loss :  4.3425934848073666e-08
mae :  0.00019666850857902318
RMSE :  0.00020838890033362997
R2 :  0.9999999999450533
verbose==0 : 
loss :  3.292345240879513e-08
mae :  0.00016314387903548777
RMSE :  0.00018144821252875598
R2 :  0.999999999958342
verbose==1 :
loss :  3.3154776701849187e-09
mae :  4.395246651256457e-05
RMSE :  5.7580184700857975e-05
R2 :  0.9999999999958049
verbose==2 :
loss :  1.1077021966343636e-08
mae :  8.912682824302465e-05
RMSE :  0.00010524743042503351
R2 :  0.9999999999859842
verbose==3 :
loss :  1.5092356875356927e-07
mae :  0.0003520280006341636
RMSE :  0.0003884888238632737
R2 :  0.9999999998090368

'''
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

y_pred2 = model.predict(x_pred2)
print(y_pred2)
