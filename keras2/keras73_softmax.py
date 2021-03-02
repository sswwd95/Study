import numpy as np
import matplotlib.pyplot as plt

def softmax(x) :
    return np.exp(x) / np.sum(np.exp(x))

# NumPy의 np.exp() 함수는 밑(base)이 자연상수 e 인 지수함수로 변환해준다

x = np.arange(1,5)
y = softmax(x)

ratio = y
labels = y
plt.pie(ratio, labels=labels, shadow=True, startangle=90)
plt.show()

# 전부 합치면 1