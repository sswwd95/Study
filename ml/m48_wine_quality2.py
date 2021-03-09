# k86_wine_quality 파일에서 실습 후 수업!
# Y를 건드려보자!

import numpy as np
import pandas as pd

wine = pd.read_csv('../data/csv/winequality-white.csv', sep=';', index_col=None, header=0)

count_data = wine.groupby('quality')['quality'].count()
# quality의 

print(count_data)
'''
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
Name: quality, dtype: int64
'''

import matplotlib.pyplot as plt
count_data.plot()
# plt.show()

# 5~6 사이에 모여있다. 3~9까지를 상, 중, 하로 분류한다. 

# y 조절 하는 방법은 데이터를 분석할 수 있는 권한이 있어야 한다
# 캐글의 경우 카테고리 변경할 수 없기 때문에 이 방식 안된다!(그냥 확인용만 가능)