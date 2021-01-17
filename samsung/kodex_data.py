import numpy as np
import pandas as pd

ko = pd.read_csv('./samsung/csv/kodex.csv',encoding='cp949', thousands=',',index_col=0, header=0)
ko = ko.iloc[:664,[0,1,2,3,7,8]]
ko = ko.sort_index(ascending=True)
y=ko.iloc[:,0:1]
del ko['시가']
ko['시가'] =y
print(ko)
print(ko.isnull())
print(ko.isnull().sum())
ko = ko.dropna(axis=0)
print(ko)

# print(ko.corr())

# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=0.9) # 폰트크기 0.9
# sns.heatmap(data=ko.corr(), square=True, annot=True, cbar=True)
# # sns.heatmap(data=df.corr(), square=정사각형으로, annot=글씨 , cbar=오른쪽에 있는 bar)
# plt.show()

np.save('./samsung/npy/ko_data.npy',arr=ko)


# [664 rows x 6 columns]