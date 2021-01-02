import numpy as np

from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape) # (506, 13)
print(y.shape)  # (506, )
print("=================")
print(x[:5])  # 0~4까지  -> x 1개당 [6.3200e-03 1.8000e+01 2.3100e+00 0.0000e+00 5.3800e-01 
#                                   6.5750e+00 6.5200e+01 4.0900e+00 1.0000e+00 2.9600e+02 1.5300e+01 3.9690e+02 4.9800e+00] 13개씩 들어있음. 
print(y[:10])

print(np.max(x), np.min(x)) # 711.0  0,0
print(dataset.feature_names)
print(dataset.DESCR)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, shuffle = True)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape = (13,)))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, batch_size = 1, epochs=100, validation_split=0.2)


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


# loss :  18.886865615844727
# mae :  3.0773253440856934
# RMSE :  4.345902354661871
# R2 :  0.7767368669007811