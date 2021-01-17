import numpy as np
import pandas as pd

ko = np.load('./samsung/npy/ko_data.npy',allow_pickle='True')

print(ko.shape) #(664, 6)
print(ko)
print(type(ko)) #<class 'numpy.ndarray'>

def split_x(seq, size, col):
    aaa=[]
    for i in range(len(seq)-size+1):
        subset=seq[i:(i+size),0:col].astype('float32')
        aaa.append(subset)
    return np.array(aaa)

size=5
col=6

ko=split_x(ko, size, col)

print(ko.shape) #(660, 5, 6)


x1 = ko[:-1,:,:-1] 
print(x1.shape) #(659, 5, 5)
y1 = ko[1:,-1:,-1:] 
print(y1.shape) #(659, 1, 1)
x1_pred = ko[-1:,:,:-1]
print(x1_pred.shape) #(1, 5, 5)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1, y1, train_size = 0.8, random_state=50)
x1_train, x1_val, y1_train, y1_val = train_test_split(
    x1_train, y1_train, train_size=0.8, random_state=50)


x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1]*x1_train.shape[2])
x1_test = x1_test.reshape(x1_test.shape[0], x1_test.shape[1]*x1_test.shape[2])
x1_pred = x1_pred.reshape(x1_pred.shape[0], x1_pred.shape[1]*x1_pred.shape[2])
x1_val =  x1_val.reshape(x1_val.shape[0], x1_val.shape[1]*x1_val.shape[2])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x1_train)
x1_train = scaler.transform(x1_train)
x1_test = scaler.transform(x1_test)
x1_pred = scaler.transform(x1_pred)
x1_val = scaler.transform(x1_val)

print(x1_train.shape) #(421, 25)
print(x1_test.shape) # (132, 25)
print(x1_pred.shape) #(1,25)
print(x1_val.shape)#(106, 25)

x1_train = x1_train.reshape(x1_train.shape[0],5,5)
x1_test = x1_test.reshape(x1_test.shape[0],5,5)
x1_pred = x1_pred.reshape(x1_pred.shape[0],5,5)
x1_val =  x1_val.reshape(x1_val.shape[0], 5,5)


y1_train = y1_train.reshape(y1_train.shape[0],1)
y1_test = y1_test.reshape(y1_test.shape[0],1)

print(y1_train.shape)

np.save('./samsung/npy/ko.npy',arr=[x1_train, x1_test, x1_val, y1_train, y1_test,y1_val,x1_pred])
