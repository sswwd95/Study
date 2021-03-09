from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

x, y = load_boston(return_X_y= True) # return_X_y : x, y 바로 분리되어 나온다

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = True, random_state = 66
)

model = XGBRegressor(n_jobs = 8)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2 : ', score)

thresholds = np.sort(model.feature_importances_) # 이렇게 하면 fi 값들이 정렬되어 나온다!
print(thresholds)
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358]

# 각각의 피쳐마다 중요도가 있는데, 중요도 낮은순으로 하나씩 빼면서 포문을 돌린거다!!
# 모두 더하면 1
# 결과물 출력시 오히려 4개의 피쳐를 뺀 결과가 제일 좋았다.

# 과제1. prefit 에 대해서 알아보기


# 하나씩 포문돌린다
# selectfrommodel : 중요도 가중치를 기반으로 기능을 선택하기 위한 메타 트랜스포머
# Xgbooster, LGBM, RandomForest등 feature_importances_기능을 쓰는 모델이면 사용 가능

for thresh in thresholds:
    selection = SelectFromModel(model, threshold = thresh, prefit = True)
    # threshold = thresh : feature 선택에 사용할 임계 값
    # prefit = true  : 사전 맞춤 모델이 생성자에 직접 전달 될 것으로 예상,
    # True 인 경우 transform직접 호출 

    select_x_train = selection.transform(x_train) # x_train을 selection형태로 바꿈
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs = 13)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, select_x_train.shape[1], score*100))
'''
(404, 13)   -> 13개 컬럼을 다 사용했을 때
Thresh=0.001, n=13, R2: 92.21%
(404, 12)   -> 가장 작은 컬럼을 뺐을 때
Thresh=0.004, n=12, R2: 92.16%
(404, 11)
Thresh=0.012, n=11, R2: 92.03%
(404, 10)
Thresh=0.012, n=10, R2: 92.19%
(404, 9)     -> 컬럼 4개를 뺐을 때가 가장 R2높다.
Thresh=0.014, n=9, R2: 93.08%
(404, 8)
Thresh=0.015, n=8, R2: 92.37%
(404, 7)
Thresh=0.018, n=7, R2: 91.48%
(404, 6)
Thresh=0.030, n=6, R2: 92.71%
(404, 5)
Thresh=0.042, n=5, R2: 91.74%
(404, 4)
Thresh=0.052, n=4, R2: 92.11%
(404, 3)
Thresh=0.069, n=3, R2: 92.52%
(404, 2)
Thresh=0.301, n=2, R2: 69.41%
(404, 1)
Thresh=0.428, n=1, R2: 44.98%
'''

# m44 기울기, 편향 부분 여기서 확인

print(model.coef_)
print(model.intercept_)
# AttributeError: Coefficients are not defined for Booster type None
# weight가 없는게 아니다. 유사개념은 다 있음