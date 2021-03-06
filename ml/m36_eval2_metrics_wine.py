# 다중분류 metric을 3개 이상 넣어서 학습

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
# 회귀

# 1. 데이터
x, y = load_wine(return_X_y=True) # 아래와 동일한 방식
# datasets = load_boston()
# x = datasets.data
# y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=True, random_state=77
)

# 2. 모델
model = XGBClassifier(n_estimators=100, learning_rate=0.01, n_jobs=8)

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric=['mlogloss','merror','auc'],
        eval_set=[(x_train, y_train),(x_test, y_test)])
aaa = model.score(x_test, y_test)
print('aaa: ',aaa)

y_pred = model.predict(x_test)
# r2 = r2_score(y_test, y_pred) # 스코어 잡을 때 기존데이터가 앞에 들어가야한다. 
# r2 = r2_score(y_pred, y_test) # 값 다르게 나온다. 
# print('r2 : ' ,r2)
# aaa:  0.9078482289043561
# r2 :  0.9078482289043561

# results = model.evals_result()
# print(results)
# 'validation_0': OrderedDict([('rmse', [23.969549, 23.741985, 23.516665,
# 숫자가 줄어드는게 보인다. 이 숫자의 개수는 n_estimators와 동일

# aaa:  0.9444444444444444