
# pca 0.95이상
# gs, rs 둘 다 해보기

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier,XGBRegressor, plot_importance
from sklearn.pipeline import Pipeline, make_pipeline 
import matplotlib.pyplot as plt
import datetime
import time
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(-1,x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(-1,x_test.shape[1]*x_test.shape[2])

pca = PCA(n_components=154)
x2_train = pca.fit_transform(x_train)
x2_test = pca.fit_transform(x_test)

x_train, x_test, y_train, y_test = train_test_split(
    x2_train, y_train, random_state=77, train_size=0.8
)

parameters = [
    {'a__n_estimators': [100,200,300], 'a__learning_late' : [0.1,0.01,0.001],
    'a__max_depth' : [6,8,10], 'a__colsample_bytree' : [0.6,0.8,1]},
    {'a__n_estimators': [100,200,300], 'a__learning_late' : [0.1,0.01,0.001],
    'a__max_depth' : [6,8,10], 'a__colsample_bylevel' : [0.6,0.8,1]}
]

pipe = Pipeline([('scaler', MinMaxScaler()),('a', XGBRegressor(n_jobs=-1, use_label_encoder=False))])


start = time.time()

# 2. 모델구성
model = GridSearchCV(pipe, parameters, cv=5)
# model = RandomizedSearchCV(pipe, parameters, cv=5)


# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('기존 acc:', acc)

sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print("작업 시간 : ", times) #작업 시간 :  0:01:03


# mnist_cnn
# loss, acc :  0.06896616518497467 0.9800999760627747

# mnist_dnn
# loss, acc :  0.13697706162929535 0.9828000068664551

# mnist_pca(0.85이상)
# loss, acc :  0.1388961225748062 0.9789999723434448

# xgb (default)
# acc :  0.9635833333333333
# 작업 시간 :  198.1862232685089

#xgb(파라미터 튜닝 후)