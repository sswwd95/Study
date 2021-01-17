import numpy as np
import pandas as pd

sam = np.load('./samsung/npy/samsung3_data.npy',allow_pickle='True')

print(sam.shape) #(664, 6)
print(sam)
print(type(sam)) #<class 'numpy.ndarray'>

def split_x(seq, size, col):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size),0:col].astype('float32')
        aaa.append(subset)
    return np.array(aaa)

size=5
col=6

sam=split_x(sam, size, col)
print(sam.shape) #(660, 5, 6)

print(sam)
x = sam[:-1,:,:-1] 
print(x.shape) #(659, 5, 5)
y = sam[1:,-1:,-1:] 
print(y.shape) #(659, 1, 1)
x_pred = sam[-1:,:,:-1]
print(x_pred.shape) #(1, 5, 5)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state=50)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=0.8, random_state=50)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1]*x_pred.shape[2])
x_val =  x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],5,5)
x_test = x_test.reshape(x_test.shape[0],5,5)
x_pred = x_pred.reshape(x_pred.shape[0],5,5)
x_val =  x_val.reshape(x_val.shape[0], 5,5)

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

print(y_train.shape)

np.save('./samsung/npy/sam.npy',arr=[x_train, x_test, x_val, y_train, y_test,y_val, x_pred])
