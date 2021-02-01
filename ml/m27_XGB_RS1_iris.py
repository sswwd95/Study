'''

parameters = [
    {'n_estimators' : [100,200,300], 'learning_late' : [0.1,0.01],'max_depth' : [6,8,10,12]},
    {'n_estimators' : [100,200,300],'max_depth' : [6,8,10,12],'learning_late' : [0.1,0.01], 'colsample_bytree':[0.6,0.9,1]},
    {'n_estimators' : [100,200,300],'max_depth' : [6,8,10,12],'learning_late' : [0.1,0.01], 'colsample_bytree':[0.6,0.9,1], 'colsample_bylevel':[0.6,0.7,0.9]},

]
n_jobs =-1
'''

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier,XGBRegressor, plot_importance
from sklearn.pipeline import Pipeline, make_pipeline 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, random_state=77, train_size=0.8
)

parameters = [
    {'a__n_estimators': [100,200,300], 'a__learning_late' : [0.1,0.01,0.001],
    'a__max_depth' : [6,8,10], 'a__colsample_bytree' : [0.6,0.8,1]},
    {'a__n_estimators': [100,200,300], 'a__learning_late' : [0.1,0.01,0.001],
    'a__max_depth' : [6,8,10], 'a__colsample_bylevel' : [0.6,0.8,1]}
]

pipe = Pipeline([('scaler', MinMaxScaler()),('a', XGBRegressor(n_jobs=-1))])

# 2. 모델구성
model = GridSearchCV(pipe, parameters, cv=5)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('기존 acc:', acc)

# plot_importance(model)
# plt.show()

# 기존 acc: 0.8947844251085385
