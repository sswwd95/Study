# 실습
# outliers1을 행렬형태로 적용할 수 있도록 수정

import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
                [100,200,3,400,500,600,700,8,900,1000]])
aaa = aaa.transpose()

print(aaa.shape) #(10, 2)

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25,50,75])
    # 데이터의 25% 지점, 50% 지점, 75% 지점
    print("1사분위(25% 지점) : ",quartile_1)
    print("q2(50% 지점) : ", q2)
    print("3사분위(75% 지점) : ", quartile_3)
    iqr = quartile_3 - quartile_1 # (3사분위 - 1사분위)를 기본값으로 가진다
    print('iqr : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5) # 통상적으로 이상치를 1.5 범위를 제일 많이 준다.
    upper_bound = quartile_3 + (iqr * 1.5)
    print("lower_bound : ", lower_bound)
    print("upper_bound : ", upper_bound)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc1 = outliers(aaa[:,0])
print("이상치의 위치 : ", outliers_loc1)
print('===================================')
outliers_loc2 = outliers(aaa[:,1])
print("이상치의 위치 : ", outliers_loc2)

'''
1사분위(25% 지점) :  3.25
q2(50% 지점) :  6.5
3사분위(75% 지점) :  97.5
iqr :  94.25
lower_bound :  -138.125
upper_bound :  238.875
이상치의 위치 :  (array([4, 7], dtype=int64),)
===================================
1사분위(25% 지점) :  125.0
q2(50% 지점) :  450.0
3사분위(75% 지점) :  675.0
iqr :  550.0
lower_bound :  -700.0
upper_bound :  1500.0
이상치의 위치 :  (array([], dtype=int64),)
'''