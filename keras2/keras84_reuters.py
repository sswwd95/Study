from tensorflow.keras.datasets import reuters # 로이터 뉴스 기사 데이터
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=30000, test_split=0.2)

print(x_train[0],type(x_train[0]))
'''
[1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 
19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 
155, 11, 15, 7, 48, 9, 4579, 1005, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 
90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12] <class 'list'>
'''
print(y_train[0]) #3
print(len(x_train[0]),len(x_train[11])) #87 59 -> 문장마다 길이가 다르다
print('++++++++++++++++++++++++++++')
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# (8982,) (2246,)
# (8982,) (2246,)

print('뉴스기사 최대길이 : ', max(len(i) for i in x_train))
# 뉴스기사 최대길이 :  2376
print('뉴스기사 평균길이 : ', sum(map(len, x_train)) / len(x_train))
# 뉴스기사 평균길이 :  145.5398574927633

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

# y분포
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
print('y분포 : ', dict(zip(unique_elements, counts_elements)))
# y분포 :  {0: 55, 1: 432, 2: 74, 3: 3159, 4: 1949, 5: 17, 6: 48,
#  7: 16, 8: 139, 9: 101, 10: 124, 11: 390, 12: 49, 13: 172, 14: 26,
#  15: 20, 16: 444, 17: 39, 18: 66, 19: 549, 20: 269, 21: 100, 22: 15, 
# 23: 41, 24: 62, 25: 92, 26: 24, 27: 15, 28: 48, 29: 19, 30: 45, 
# 31: 39, 32: 32, 33: 11, 34: 50, 35: 10, 36: 49, 37: 19, 38: 19,
#  39: 24, 40: 36, 41: 30, 42: 13, 43: 21, 44: 12, 45: 18}

# plt.hist(y_train, bins=46)
# plt.show()

# x의 단어들 분포
word_to_index = reuters.get_word_index()
print(word_to_index)
'''
{'mdbl': 10996, 'fawc': 16260, 'degussa': 12089, 'woods': 8803, 
'hanging': 13796, 'localized': 20672, 'sation': 20673, 
'chanthaburi': 20675, 'refunding': 10997, 'hermann': 8804,
 'passsengers': 20676, 'stipulate': 20677, 'heublein': 8352, 
 'screaming': 20713, 'tcby': 16261, 'four': 185, 'grains': 1642, 
 'broiler': 20680, 'wooden': 12090, 'wednesday': 1220,
'''
print(type(word_to_index)) #<class 'dict'>

# 키와 벨류를 교체!
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

# 키 벨류 교환 후 
print(index_to_word)
# {10996: 'mdbl', 16260: 'fawc', 12089: 'degussa',
#  8803: 'woods', 13796: 'hanging', 20672: 'localized',
#  20673: 'sation', 20675: 'chanthaburi', 10997: 'refunding',
#  8804: 'hermann', 20676: 'passsengers', 20677: 'stipulate',
#  8352: 'heublein', 20713: 'screaming', 16261: 'tcby',
#  185: 'four', 1642: 'grains', 20680: 'broiler', 12090: 'wooden', 
# 1220: 'wednesday',

print(index_to_word[1]) #the -> 가장 빈도수가 많은 단어
print(len(index_to_word)) #30979
print(index_to_word[30979]) #northerly -> 가장 빈도수가 적은 단어

# x_train[0]
print(x_train[0])
print(' '.join([index_to_word[index] for index in x_train[0]]))
'''
[1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 
19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 
155, 11, 15, 7, 48, 9, 4579, 1005, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245,
90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12]
the of of mln loss for plc said at only ended said commonwealth could 1 
traders now april 0 a after said from 1985 and from foreign 000 april 0 
prices its account year a but in this mln home an states earlier and rise
 and revs vs 000 its 16 vs 000 a but 3 psbr oils several and shareholders 
 and dividend vs 000 its all 4 vs 000 1 mln agreed largely april 0 are 2 
 states will billion total and against 000 pct dlrs
'''
# 값이 이상하다. 왜? num_words = 10000으로 줘서 나머지 단어 잘렸기 때문

# num_words = 30000으로 하면
'''
[1, 27595, 28842, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 
102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 
15, 22, 155, 11, 15, 7, 48, 9, 4579, 1005, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 
197, 1245, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12]
the wattie nondiscriminatory mln loss for plc said at only ended said commonwealth could 1 traders now 
april 0 a after said from 1985 and from foreign 000 april 0 prices its account year a but in this mln 
home an states earlier and rise and revs vs 000 its 16 vs 000 a but 3 psbr oils several and shareholders
 and dividend vs 000 its all 4 vs 000 1 mln agreed largely april 0 are 2 states will billion total and against 000 pct dlrs
'''

