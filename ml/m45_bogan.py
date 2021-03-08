# 결측치 처리 할 때 
# np.nan 발견할 때 0으로 처리, 중간값 넣기, 평균값 넣기, 행 전체 삭제 가능, bogan 가능

'''
x = 1, 2, nan, 4, 5, nan, 7, 8, 9, 10
y = 1, 2, nan, 4, 5, nan, 7, 8, 9, 10
x = (1,2,4,5,7,8,9,10)
x_train = (1,2,4,5,7,8)
x_test = (9,10)

model = mdoel.아무거나()
model.fit(x_train, y_train)

model.predict([Nan Nan]) Nan 데이터만 빼서 프레딕트 한다. 그럼 결측치의 빈 값이 나온다.
Nan 값이 나오면 다시 모델에 넣어서 돌린다

시계열에서 잘 먹힌다. 
'''

from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datestrs = ['3/1/2021', '3/2/2021','3/3/2021','3/4/2021','3/5/2021',]
dates = pd.to_datetime(datestrs)
print(dates)
# DatetimeIndex(['2021-03-01', '2021-03-02', '2021-03-03', '2021-03-04',
#                '2021-03-05'],
#               dtype='datetime64[ns]', freq=None)
print("=================================")

ts = Series([1,np.nan, np.nan, 8, 10], index=dates)
print(ts)
'''
2021-03-01     1.0
2021-03-02     NaN
2021-03-03     NaN
2021-03-04     8.0
2021-03-05    10.0
dtype: float64
'''

ts_intp_linear = ts.interpolate()
# 조건이 가급적이면 시계열에서 쓰는게 좋다. 연속된 데이터에 잘 맞다.
# 결측치가 알아서 수정이 된다. 
print(ts_intp_linear)
'''
2021-03-01     1.000000
2021-03-02     3.333333
2021-03-03     5.666667
2021-03-04     8.000000
2021-03-05    10.000000
dtype: float64
'''
