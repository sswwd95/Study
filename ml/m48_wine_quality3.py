# k86_wine_quality 파일에서 실습 후 수업!

import numpy as np
import pandas as pd


############################################ 기존 코딩 방법 ############################################
# -----새로운 WINE 파일로 해보자!-----------
wine = pd.read_csv('../data/csv/winequality-white.csv', sep=';', index_col=None, header=0)
wine_npy = wine.to_numpy() # == wine.values

y = wine['quality'] # y = wine_npy[:,-1]
x = wine.drop('quality', axis=1) # x = wine_npy[:,:-1]

print(x.shape, y.shape)
# (4898, 11) (4898,)


#--------------------- y 조절하기(임의로 가능) -----------------------
newlist = []
for i in list(y):
    if i <= 4: # i는 0번째부터 들어간다. (3,4) => 0등급
        newlist +=[0]
    elif i <= 7:  #(5,6,7) => 1등급
        newlist +=[1]
    else :        # (8, 9) => 2등급 표본은 임의로 잡는다
        newlist +=[2]
y = newlist
#--------------------------------------------------------------------


from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

print(x_train.shape, x_test.shape)
# (3918, 11) (980, 11)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

model = RandomForestClassifier()

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('score : ', score)


# elif i <= 7:  
# score :  0.9520408163265306

# elif i <= 6:  
# score :  0.8581632653061224


