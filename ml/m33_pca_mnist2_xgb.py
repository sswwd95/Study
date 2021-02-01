# m31로 만든 1.0이상의 n_component=?를 사용하여 xgb 모델 만들기
# mnist 0.95 =154, 1.0 = 713
# mnist dnn보다 성능 좋게 만들것
# cnn과 비교

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier
from tensorflow.keras.datasets import mnist

import datetime
import time


#1. 데이터
(x_train,y_train), (x_test,y_test) = mnist.load_data()
# x = np.append(x_train, x_test, axis=0)
# print(x.shape) #(70000, 28, 28)

# ->  2차원으로 reshape
x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])

pca = PCA(n_components=713)
x2_train = pca.fit_transform(x_train)
x2_test = pca.fit_transform(x_test)

x_train, x_test,y_train, y_test= train_test_split(x2_train, y_train, train_size=0.8, random_state=77 )

start = time.time()
#2. 모델
model = XGBClassifier(n_jobs=-1, use_label_encoder=False)

#3. 훈련
model.fit(x_train, y_train,eval_metric='logloss')

#4. 평가 예측
acc = model.score(x_test,y_test)
print('acc : ', acc)

sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]

print("작업 시간 : ", times) 

# acc :  0.9579166666666666
# 작업 시간 :  892.8674330711365




