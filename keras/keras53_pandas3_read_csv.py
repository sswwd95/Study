import numpy as np
import pandas as pd

df = pd.read_csv('../data/csv/iris_sklearn.csv')

print(df) # 인덱스가 데이터로 들어감. 
'''
     Unnamed: 0  sepal_length  sepal_width  petal_length  petal_width  Target
0             0           5.1          3.5           1.4          0.2       0
1             1           4.9          3.0           1.4          0.2       0
2             2           4.7          3.2           1.3          0.2       0
3             3           4.6          3.1           1.5          0.2       0
4             4           5.0          3.6           1.4          0.2       0
..          ...           ...          ...           ...          ...     ...
145         145           6.7          3.0           5.2          2.3       2
146         146           6.3          2.5           5.0          1.9       2
147         147           6.5          3.0           5.2          2.0       2
148         148           6.2          3.4           5.4          2.3       2
149         149           5.9          3.0           5.1          1.8       2

[150 rows x 6 columns]
'''

df = pd.read_csv('../data/csv/iris_sklearn.csv',index_col=0)

print(df) # 헤더가 자동으로 표시됨. 헤더 없는 데이터면 데이터 첫번째 줄을 헤더로 인식해버림
'''
     sepal_length  sepal_width  petal_length  petal_width  Target
0             5.1          3.5           1.4          0.2       0
1             4.9          3.0           1.4          0.2       0
2             4.7          3.2           1.3          0.2       0
3             4.6          3.1           1.5          0.2       0
4             5.0          3.6           1.4          0.2       0
..            ...          ...           ...          ...     ...
145           6.7          3.0           5.2          2.3       2
146           6.3          2.5           5.0          1.9       2
147           6.5          3.0           5.2          2.0       2
148           6.2          3.4           5.4          2.3       2
149           5.9          3.0           5.1          1.8       2

[150 rows x 5 columns]
'''

df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0)
# 기본값 : index_col = None, header = 1 / header=None 은 칼럼 이름이 없다는 뜻이며, 만약 1번째 행이 칼럼 이름이라면 header=0 으로 지정
print(df) 






'''
#######################################################################
# to_csv에서 read_csv까지 한 파일에 만들기 가능
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
x = dataset['data']
y = dataset['target']

df = pd.DataFrame(x, columns=dataset['feature_names']) 

# 컬럼 이름 변경
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# y 컬럼 추가
df['Target'] = y 

# dataframe을 csv로 만들기
df.to_csv('../data/csv/iris_sklearn.csv',sep=',')

# read_csv
df = pd.read_csv('../data/csv/iris_sklearn.csv')
df = pd.read_csv('../data/csv/iris_sklearn.csv', index_col=0, header=0)
print(df) 
#########################################################################
'''
