import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)
y = np.tanh(x)

plt.plot(x,y)
plt.grid()
plt.show()

#-1과 1사이로 수렴. LSTM 내부에 있다.