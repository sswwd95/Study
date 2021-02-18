import p71_byunsu as p71
print(p71.aaa) #2
print(p71.square(10)) #1024 (제곱)

print("___________________________________")

from p71_byunsu import aaa, square

aaa=3
print(aaa) #3
# 다른 메모리 속에 있는 값이다. 가장 가까운 값을 불러온다.

print(square(10)) #1024 (제곱) -> import 했던 값이 나온다.(p71.aaa =2)
