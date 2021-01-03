#실습. train_size / test_size에 대해 완벽 이해할 것

from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(1,101))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,
                             #train_size=0.9, test_size=0.2, shuffle=True)  
                            train_size = 0.7, test_size=0.2,shuffle=True)
                             # 위 두가지의 경우에 대해 확인 후 정리할 것!
                             # 1.1 이 넘을 경우와 1이 안될 경우 비교 
                             # 1.1 이 넘으면 => The sum of test_size and train_size = 1.1, should be in the (0, 1) range. Reduce test_size and/or train_size.
                             # 1.1 이 안넘으면 => [53 37 16 41 77 65 54 27 45 24 57 93 75 13 71 97 76  9 20 40 38 78  2 66
                                                # 64 15 58 69 21 86 29 81  1 63 39 10 59 62 89 36 51 72 14  3 44 52 26 82
                                                # 6 80 35 95 87 19 17 25 11 92 60 98 67 61 74 18 28 99  8 47 96 83]
                                                # (70,)
                                                # (20,)
                                                # (70,)
                                                # (20,)
print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
