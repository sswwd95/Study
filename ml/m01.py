# 1.27 머신러닝 시작!!

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1) # 0~10까지 0.1 단위로 나누는 것. 100개 생성.
y = np.sin(x)

plt.plot(x,y)
plt.show()

# 그냥 sin 그래프 그려봄.