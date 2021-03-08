# 1, 2, 3, 4, 5, 6, 7, 8, 9, 10만 이상치 처리 어떻게 가능?
# 삭제하면 데이터 나간다. 
# 10만을 nan으로 바꾸고 bogan으로 처리
# 0으로 처리
# 판단은 컬럼으로 하고 삭제는 행 전체.
# lsjsj92.tistory.com/556

import numpy as np
aaa = np.array([1,2,3,4,6,7,90,100,5000,10000])
# 2와 3사이가 (1사분위=25%), 6과 7 사이가 중간지점, 75%(3분위)지점은 90과 100사이 쯤?

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

outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)

# 1사분위(25% 지점) :  3.25
# q2(50% 지점) :  6.5
# 3사분위(75% 지점) :  97.5
# 이상치의 위치 :  (array([8, 9], dtype=int64),)
# iqr :  94.25
# lower_bound :  -138.125
# upper_bound :  238.875
# 정상적인 값의 범위가 lower_bound ~ upper_bound 라고 평가한다.
# 이상 넘어가면 이상치라고 판단
# 평균값을 이용했다면 이런 결과가 나올 수 없다. 

# 왜 1.5 곱해주나요? 임의로 해준 것

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

# 이 함수의 문제점? 이거는 한개밖에 못 써(보통 다차원)