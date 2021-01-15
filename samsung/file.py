import numpy as np
import pandas as pd

df = pd.read_csv('./samsung/samsung.csv',thousands=',',index_col=0,header=0)
df = df.iloc[:-1]
df['Target'] = df.iloc[:,3]
del df['종가']
data = pd.read_csv('./samsung/samsung2.csv',encoding='cp949',thousands=',',index_col=0,header=0)
data['Target'] = data.iloc[:,3]
del data['종가']
print(data)
print(data.info())
df2 = pd.concat([data,df])
print(df2)  
print(df2.isnull().sum())

print(df.corr())
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=0.9) # 폰트크기 0.9
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()

