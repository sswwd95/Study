gradient = lambda x:2*x - 4
# x를 넣었을 때 2*x - 4 를 반환 (weight =2, bias = -4)

def gradient2(x) : 
    temp = 2*x -4
    return temp

x = 3

print(gradient(x))
print(gradient2(x))