import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset['data'] # => x = dataset.data
y = dataset['target'] # => y = dataset.target

df = pd.DataFrame(x, columns=dataset['feature_names']) 

# # 컬럼 명 변경
# df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# # y 칼럼 추가
# df['Target'] = y  # -> df['Target'] = dataset.target

# dataframe을 csv로 만들기
df.to_csv('../data/csv/diabetes_sklearn.csv',sep=',')
# 엑셀로 확인가능, vscode로도 확인 가능
# market에서 rainbow csv설치하면 색 표시, edit csv 설치하면 엑셀처럼 보임
