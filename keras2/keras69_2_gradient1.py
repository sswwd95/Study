import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x +6 # 2차함수
x = np.linspace(-1,6,100) #-1부터 6까지 , 그 사이에 100개 들어있다.
y = f(x)

# 그리기
plt.plot(x,y,'k-') # k- : 색 표현
plt.plot(2,2,'sk')  # sk : 점 찍는 것 
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

'''
plt.plot(x,y,'y-')
b = blue, g= green, r= red, c =cyan, m = magenta, y = yellow, k = black, w = white
https://cometouniverse.tistory.com/28
'''