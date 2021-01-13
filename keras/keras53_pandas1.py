import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset.keys())
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(dataset.values())
# dict_values([array([[5.1, 3.5, 1.4, 0.2],
#        [4.9, 3. , 1.4, 0.2],
#        [4.7, 3.2, 1.3, 0.2],
#        [4.6, 3.1, 1.5, 0.2],
#        ..
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
print(dataset.filename)
# C:\Users\bit\Anaconda3\lib\site-packages\sklearn\datasets\data\iris.csv -> 파일경로
print(dataset.target_names)
# ['setosa' 'versicolor' 'virginica'] -> 0,1,2

# 둘 다 똑같은 표현
x = dataset['data'] # => x = dataset.data
y = dataset['target'] # => y = dataset.target
print(x)
print(y)
print(x.shape,y.shape) # (150, 4) (150,)
print(type(x), type(y)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# x값을 dataframe에 넣는다

df = pd.DataFrame(x, columns=dataset['feature_names']) # df = pd.DataFrame(x, columns=dataset.feature_names)

print(df) # df는 변수명. 임의로 변경가능
'''
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)   -> header = column. 헤더는 데이터가 아님
0                  5.1               3.5                1.4               0.2
1                  4.9               3.0                1.4               0.2
2                  4.7               3.2                1.3               0.2
3                  4.6               3.1                1.5               0.2
4                  5.0               3.6                1.4               0.2
..                 ...               ...                ...               ...
145                6.7               3.0                5.2               2.3
146                6.3               2.5                5.0               1.9
147                6.5               3.0                5.2               2.0
148                6.2               3.4                5.4               2.3
149                5.9               3.0                5.1               1.8
-> 앞의 숫자들은 index. 데이터 아님
[150 rows x 4 columns]
'''
print(df.shape) #(150, 4) pandas에서만 먹히는 것. 원래는 list에서 shape 안된다
print(df.columns) #Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)'],dtype='object')
print(df.index) # RangeIndex(start=0, stop=150, step=1) 
# 명시 안하면 자동으로 인덱싱된다. 

print(df.head()) # = df[:5]
'''
  sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2
'''
print(df.tail()) # = df[-5:]
'''
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
145                6.7               3.0                5.2               2.3
146                6.3               2.5                5.0               1.9
147                6.5               3.0                5.2               2.0
148                6.2               3.4                5.4               2.3
149                5.9               3.0                5.1               1.8
'''
print(df.info())
'''
RangeIndex: 150 entries, 0 to 149
Data columns (total 4 columns):
 #   Column             Non-Null Count  Dtype
---  ------             --------------  -----
 0   sepal length (cm)  150 non-null    float64   -> non-null : null값이 없다는 것. 결측치(데이터에 값이 없는 것) 없다. 
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64
dtypes: float64(4)
memory usage: 4.8 KB
None
'''
print(df.describe())
'''
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
count         150.000000        150.000000         150.000000        150.000000
mean(평균)       5.843333          3.057333           3.758000          1.199333
std (표준편차)   0.828066          0.435866           1.765298          0.762238
min             4.300000          2.000000           1.000000          0.100000
25%             5.100000          2.800000           1.600000          0.300000
50%             5.800000          3.000000           4.350000          1.300000
75%             6.400000          3.300000           5.100000          1.800000
max             7.900000          4.400000           6.900000          2.500000
'''
############
#컬럼명 수정
############
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
print(df.columns) # Index(['sepal_lenth', 'sepal_width', 'petal_length', 'petal_width'], dtype='object')
print(df.info())
'''
RangeIndex: 150 entries, 0 to 149
Data columns (total 4 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   sepal_length   150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
dtypes: float64(4)
memory usage: 4.8 KB
None
'''
print(df.describe())
'''
      sepal_length  sepal_width  petal_length  petal_width
count   150.000000   150.000000    150.000000   150.000000
mean      5.843333     3.057333      3.758000     1.199333
std       0.828066     0.435866      1.765298     0.762238
min       4.300000     2.000000      1.000000     0.100000
25%       5.100000     2.800000      1.600000     0.300000
50%       5.800000     3.000000      4.350000     1.300000
75%       6.400000     3.300000      5.100000     1.800000
max       7.900000     4.400000      6.900000     2.500000
'''
##################
# y 컬럼을 추가하기
##################
print(df['sepal_length'])
df['Target'] = dataset.target
print(df.head())
'''
   sepal_length  sepal_width  petal_length  petal_width  Target
0           5.1          3.5           1.4          0.2       0
1           4.9          3.0           1.4          0.2       0
2           4.7          3.2           1.3          0.2       0
3           4.6          3.1           1.5          0.2       0
4           5.0          3.6           1.4          0.2       0
'''

print(df.shape) # (150, 5)
print(df.columns) # Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Target'], dtype='object')
print(df.index) # 동일함
print(df.tail())
'''
     sepal_length  sepal_width  petal_length  petal_width  Target
145           6.7          3.0           5.2          2.3       2
146           6.3          2.5           5.0          1.9       2
147           6.5          3.0           5.2          2.0       2
148           6.2          3.4           5.4          2.3       2
149           5.9          3.0           5.1          1.8       2
'''
print(df.info())
'''
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   Target        150 non-null    int32
dtypes: float64(4), int32(1)
memory usage: 5.4 KB
None
'''
print(df.isnull()) # null이 있는지 확인
'''
     sepal_length  sepal_width  petal_length  petal_width  Target
0           False        False         False        False   False
1           False        False         False        False   False
2           False        False         False        False   False
3           False        False         False        False   False
4           False        False         False        False   False
..            ...          ...           ...          ...     ...
145         False        False         False        False   False
146         False        False         False        False   False
147         False        False         False        False   False
148         False        False         False        False   False
149         False        False         False        False   False
[150 rows x 5 columns]
'''
print(df.isnull().sum()) # null이 몇 개 있는지 확인
'''
sepal_length    0
sepal_width     0
petal_length    0
petal_width     0
Target          0
dtype: int64
'''
print(df.describe())
'''
       sepal_length  sepal_width  petal_length  petal_width      Target
count    150.000000   150.000000    150.000000   150.000000  150.000000
mean       5.843333     3.057333      3.758000     1.199333    1.000000
std        0.828066     0.435866      1.765298     0.762238    0.819232
min        4.300000     2.000000      1.000000     0.100000    0.000000
25%        5.100000     2.800000      1.600000     0.300000    0.000000
50%        5.800000     3.000000      4.350000     1.300000    1.000000
75%        6.400000     3.300000      5.100000     1.800000    2.000000
max        7.900000     4.400000      6.900000     2.500000    2.000000
'''

print(df['Target'].value_counts())  # y의 value값 몇 개인지 확인
'''
2    50   -> 2가 50개 
1    50   -> 1이  //
0    50   -> 0이  //
Name: Target, dtype: int64
'''

###################################
# 상관계수(Correlation coefficient) 
###################################
print(df.corr())
'''
              sepal_length  sepal_width  petal_length  petal_width    Target
sepal_length      1.000000    -0.117570      0.871754     0.817941  0.782561 (sepal_length와 상관관계 많지 않은 것 같음)
sepal_width      -0.117570     1.000000     -0.428440    -0.366126 -0.426658 (sepal_width와 상관관계 적음)
petal_length      0.871754    -0.428440      1.000000     0.962865  0.949035 (petal_length와 상관관계 있음)
petal_width       0.817941    -0.366126      0.962865     1.000000  0.956547 (petal_width와 상관관계 있음)
Target            0.782561    -0.426658      0.949035     0.956547  1.000000 (target과 target은 100%)
'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=0.9) # 폰트크기 0.9
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
# sns.heatmap(data=df.corr(), square=정사각형으로, annot=글씨 , cbar=오른쪽에 있는 bar)
plt.show()

# target과 sepal_length를 선형회귀로 할 수 있다. lenear
# 시각화가 어느 정도 맞지만 100% 맞지 않음. 상관간계가 어떤지 확인하는 것. 

############
# 도수 분포도 -> 도수분포표(frequency table): 데이터를 구간으로 나누어, 각 구간의 빈도를 나타낸 표
############
plt.figure(figsize=(10,6)) #(10,6)짜리 도화지 그리기
plt.subplot(2,2,1)  #2행 2열의 첫번째
plt.hist(x='sepal_length', data=df)
# model의 hist는 loss, metrics의 history. plt의 hist는 histogram
# 히스토그램(histogram): 도수분포표를 그래프로 그린 것
plt.title('sepal_length')

plt.subplot(2,2,2)
plt.hist(x='sepal_width', data=df)
plt.title('sepal_width')

plt.subplot(2,2,3)
plt.hist(x='petal_length', data=df)
plt.title('petal_length')

plt.subplot(2,2,4)
plt.hist(x='petal_width', data=df)
plt.title('petal_width')
# plt.grid(True) # 줄 격자 표시 -> heatmap 주석처리 안하면 뒤에 예쁘게 줄 생김
plt.show()

# 그래프의 x축은 min~max 값 , y축은 x축의 값에 몇개가 속해있는지 보여줌. 다 합치면 150개
