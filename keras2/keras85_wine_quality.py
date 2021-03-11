
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #훈련 과정에서 구성한 다수의 결정 트리들을 랜덤하게 학습시켜 분류 또는 회귀의 결과도출에 사용


wine_data = pd.read_csv('../data/csv/winequality-white.csv',sep=';',dtype=float)

x_data = wine_data.iloc[:,:-1]
y_data = wine_data.iloc[:,-1]

# Score 값이 7보다 작으면 0,  7보다 크거나 같으면 1로 값 변경.
y_data = np.array([1 if i>=7 else 0 for i in y_data])

# score 값을 6 기준. 
# y_data = np.array([1 if i>=7 elif i=6 0 for i in y_data])

# 트레인, 테스트 데이터 나누기.
train_x, test_x, train_y, test_y = train_test_split(
    x_data, y_data, test_size = 0.3,random_state=42)

from sklearn.pipeline import Pipeline, make_pipeline 
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer, RobustScaler, MaxAbsScaler


parameters = [
    {'a__n_estimators': [100,200,300], 'a__learning_late' : [0.1,0.01,0.001],
    'a__max_depth' : [6,8,10], 'a__colsample_bytree' : [0.6,0.8,1]},
    {'a__n_estimators': [100,200,300], 'a__learning_late' : [0.1,0.01,0.001],
    'a__max_depth' : [6,8,10], 'a__colsample_bylevel' : [0.6,0.8,1]}
]

pipe = Pipeline([('scaler', PowerTransformer()),('a', XGBClassifier(n_jobs=-1))])

# 2. 모델구성
model = RandomizedSearchCV(pipe, parameters, cv=5)

# model = RandomForestClassifier()
model.fit(train_x, train_y)

acc = model.score(test_x, test_y)
print('acc : ', acc)

y_pred = model.predict(test_x)

from sklearn.metrics import classification_report

y_true, y_pred = test_y, model.predict(test_x)

print(classification_report(y_true, y_pred))

'''
model = RandomForestClassifier()
acc :  0.8836734693877552
              precision    recall  f1-score   support

           0       0.89      0.97      0.93      1141
           1       0.84      0.60      0.70       329

    accuracy                           0.88      1470
   macro avg       0.87      0.78      0.81      1470
weighted avg       0.88      0.88      0.88      1470
'''

'''

acc :  0.8836734693877552
              precision    recall  f1-score   support

           0       0.90      0.95      0.93      1141
           1       0.80      0.64      0.71       329

    accuracy                           0.88      1470
   macro avg       0.85      0.80      0.82      1470
weighted avg       0.88      0.88      0.88      1470
'''