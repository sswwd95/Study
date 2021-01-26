import pandas as pd
df = pd.DataFrame([[1,2,3,4],[4,5,6,7],[7,8,9,10]],
                    columns=list('abcd'), index=('가', '나', '다'))
print(df)

df2 = df #같은 메모리 공유. 새롭게 생기는 것 아님 (판다스의 경우)

df2['a'] = 100

print(df2)
print(df)

print(id(df), id(df2)) #2879073846704 2879073846704 / 동일함


df3 = df.copy() # 새롭게 생성
df2['b'] = 333
print(df)
print(df2)
print(df3)


df = df+99
print(df)
print(df2)
print(df3)


