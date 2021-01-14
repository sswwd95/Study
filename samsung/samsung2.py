import numpy as np
import pandas as pd

df = pd.read_csv('./samsung/samsung.csv',thousands=',',index_col=0,header=0)
df = df.iloc[:-1]
df = df.iloc[:662,[0,1,2,3,5,6]]
df['Target'] = df.iloc[:,3]
del df['종가']
data = pd.read_csv('./samsung/samsung2.csv',encoding='cp949',thousands=',',index_col=0,header=0)
data = data.iloc[[0],[0,1,2,3,7,8]]
data['Target'] = data.iloc[:,3]
del data['종가']
print(data)
print(data.info())
df2 = pd.concat([data,df])
print(df2)  
print(df2.isnull().sum())


data = df2.values

def split_x(seq, size, col) : 
    dataset = []
    for i in range(len(seq) - size +1) : 
        subset = seq[i:(i+size),0:col].astype('float32')
        dataset.append(subset)
    return np.array(dataset)
size = 5
col = 6

dataset = split_x(data, size, col)
print(dataset)
print(dataset.shape) #(658, 5, 6)


x = dataset[:-1,:,:-1]
print(x.shape) #(657, 5, 5)
y = dataset[1:,-1:,-1:]
print(y.shape) #(657, 1, 1)
x_pred = dataset[-1:,:,:-1]
print(x_pred.shape) #(1, 5, 5)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1]*x_pred.shape[2])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)

x = x.reshape(x.shape[0],5,5)
x_pred = x_pred.reshape(x_pred.shape[0],5,5)
print(x.shape)
print(x_pred)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = False, random_state = 66)

np.save('../data/npy/samsung2.npy',arr=[x_train, x_test, y_train, y_test,x_pred])

#2 . 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv1D(200,3,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Conv1D(100,3,activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = '../data/modelcheckpoint/samsung2_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor = 'loss', patience=20, mode='min')
model.fit(x_train, y_train, batch_size = 16, callbacks=[es, cp], epochs=1000, validation_split=0.3)

# 4. 평가 예측
loss,mae = model.evaluate(x_test, y_test, batch_size=16)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_pred)
print('1월 15일 : ', y_predict)
print(x_pred)


