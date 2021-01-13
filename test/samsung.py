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

df2 = df.iloc[:662,[0,1,2,3]]
df2 = df2.loc[::-1]
print(df2)
print(df2.shape)

ss = df2.values
print(ss)
np.save('../data/npy/samsung.npy',arr=ss)

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size +1):
        subset = seq[i : (i+size)]
        aaa.append(subset)
    return np.array(aaa)
size = 5
data = np.load('../data/npy/samsung.npy')
print(data.shape) #(662,4)
data = split_x(data,size)

x = data[:,[0,1,2]]
y = data[:,[-1]]
print(x)
print(y)
print(x.shape) #(658, 3, 4)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=10, shuffle=True
)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

#2 . 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv1D(500,3,input_shape=(x_train.shape[1],1)))
model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))


# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=20, mode='min') 

model.fit(x_train, y_train, batch_size = 32, callbacks=[es], epochs=1000, validation_split=0.2)


# 4. 평가 예측
loss,mae = model.evaluate(x_test, y_test, batch_size=32)
print("loss, mae : ", loss, mae)

y_predict = model.predict(x_pred)
print("y_predict : ",y_predict)

# RMSE, R2 = 회귀모델 지표
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

