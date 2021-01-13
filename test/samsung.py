import numpy as np
import pandas as pd
# 1. 데이터
df = pd.read_csv('./test/samsung.csv',thousands=',',index_col=0,header=0)
print(df)
print(df.shape) #(2400, 14)
print(df.info())
'''
Data columns (total 14 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   시가      2400 non-null   object
 1   고가      2400 non-null   object
 2   저가      2400 non-null   object
 3   종가      2400 non-null   object
 4   등락률     2400 non-null   float64
 5   거래량     2397 non-null   object
 6   금액(백만)  2397 non-null   object
 7   신용비     2400 non-null   float64
 8   개인      2400 non-null   object
 9   기관      2400 non-null   object
 10  외인(수량)  2400 non-null   object
 11  외국계     2400 non-null   object
 12  프로그램    2400 non-null   object
 13  외인비     2400 non-null   float64
dtypes: float64(3), object(11)
memory usage: 281.2+ KB
None
'''
# print(df.corr())

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=0.9)
# sns.heatmap(data=df.corr(),square=True, annot=True, cbar=True)

# plt.show()

df2 = df.iloc[:662, :4]
df2 = df2.loc[::-1]
print(df2)
print(df2.shape)

x = df2.iloc[:662,:3]
y = df2.iloc[:,-1:]
print(x)
print(y)
print(x.shape) #(662, 3)
print(y.shape) #(662, 1)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 50)


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

# print(x_train.shape, x_test.shape)

# # 2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Conv1D,Flatten,Dropout

# model = Sequential()
# model.add(Conv1D(300, 2, activation = 'relu', input_shape=(3,1)))
# model.add(Conv1D(200,2))
# model.add(Flatten())
# model.add(Dense(200, activation= 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(100, activation= 'relu'))
# model.add(Dense(100, activation= 'relu'))
# model.add(Dense(100, activation= 'relu'))
# model.add(Dense(100, activation= 'relu'))
# model.add(Dense(100, activation= 'relu'))
# model.add(Dense(100, activation= 'relu'))
# model.add(Dense(3, activation= 'softmax'))

# # 3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor = 'acc', patience = 20, mode='max')

# model.fit(x_train, y_train, callbacks=[early_stopping], validation_split=0.2, batch_size=8, epochs=1000)

# #. 4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test, batch_size=8)
# print('loss, acc : ', loss,acc)

# y_predict = model.predict(x_test)
# print(y_predict)
# print(np.argmax(y_predict,axis=-1))
