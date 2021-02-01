

# pca 1.0

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline, make_pipeline
import datetime, time

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1,x_train[1]*x_train[2])
x_test = x_test.reshape(-1,x_test[1]*x_test[2])

pca = PCA(n_components=713)
x2_train = pca.fit_transform(x_train)
x2_test = pca.fit_transform(x_test)

x_train, x_test, y_train, y_test = train_test_split(
    x2_train, y_train, random_state=77, train_size=0.8
)


start = time.time()

# 2. 모델구성
parameters = [
    {'n_estimators' : [100,200,300], 'learning_rate': [0.1,0.01,0.001],
    'max_depth' : [4,6,8,10], 'colsample_bytree':[0.6,0.8,1]},  
    {'n_estimators' : [100,200,300], 'learning_rate': [0.1,0.01,0.001],
    'max_depth' : [4,6,8,10], 'colsample_bytree':[0.6,0.8,1], 
    'colsample_bylevel': [0.6,0.8,1]}
]
# n_estimators = epochs

model = GridSearchCV(XGBClassifier(n_jobs=-1, use_label_encoder=False) ,parameters, cv=5)

# 훈련
model.fit(x_train, y_train, eval_metric='mlogloss', verbose=True, 
        eval_set=[(x_train, y_train), (x_test,y_test)])
# eval_metric='error' 모델.컴파일에 있는 매트릭스와 동일
# verbose를 true(=1)로 잡으면 eval_set=[(x_train, y_train), (x_test,y_test)]의 값들이 표시됨
# eval_set 안 넣으면 verbose안먹힌다. 
# 머신러닝은 verbose 디폴트가 false(표시 안됨)

# 평가, 예측
acc = model.score(x_test, y_test)
print('acc : ', acc)

sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)).split('.')
times = times[0]
print('작업시간 : ', times)

# mnist_cnn
# loss, acc :  0.06896616518497467 0.9800999760627747

# mnist_dnn
# loss, acc :  0.13697706162929535 0.9828000068664551

# mnist_pca(1.0이상)
# loss, acc :  0.24381767213344574 0.9728333353996277

# xgb(파라미터 튜닝 후)