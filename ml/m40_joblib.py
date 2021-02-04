from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
# 회귀

# 1. 데이터
# x, y = load_boston(return_X_y=True) # 아래와 동일한 방식
datasets = load_boston()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=77
)

# 2. 모델
model = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=8)

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'],
        eval_set=[(x_train, y_train),(x_test, y_test)],
        early_stopping_rounds=10)
        
aaa = model.score(x_test, y_test)
print('model.score: ',aaa)
# [99]    validation_0-rmse:9.55219       validation_0-logloss:-803.53387 validation_1-rmse:8.48393       validation_1-logloss:-752.75574
# aaa:  -0.03814756625777593
#validation 0 = train, validation1=test

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred) # 스코어 잡을 때 기존데이터가 앞에 들어가야한다. 
# r2 = r2_score(y_pred, y_test) # 값 다르게 나온다. 
print('r2 : ' ,r2)
# aaa:  0.9078482289043561
# r2 :  0.9078482289043561

# results = model.evals_result()
# print(results)
# 'validation_0': OrderedDict([('rmse', [23.969549, 23.741985, 23.516665,
# 숫자가 줄어드는게 보인다. 이 숫자의 개수는 n_estimators와 동일

# 파이썬에서 제공
import pickle
'''
############### 저장 ##########################
pickle.dump(model, open('../data/xgb_save/m39.pickle.data', 'wb'))
#dump = 소환, wb = write. 쓰겠다는것
print('저장')
################################################

################ 불러오기 #######################
model2 = pickle.load(open('../data/xgb_save/m39.pickle.data','rb'))
print('불러오기')
r22 = model2.score(x_test, y_test)
print('r22 : ', r22)
'''
import joblib # 경로명만 써주면 된다. pickle보다 간결

############### 저장 ##########################
joblib.dump(model, '../data/xgb_save/m39.joblib.data')
print('저장')
################################################

################ 불러오기 #######################
model2 = joblib.load('../data/xgb_save/m39.pickle.data')
print('불러오기')
r22 = model2.score(x_test, y_test)
print('r22 : ', r22)
