import tensorflow as tf
#compat.v1. -> 경로. 텐서플로우 1.14이상 부터 폴더 구조 바꾼 것
tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.compat.v1.random_normal([1]),name='weight')
print(W) # W에 대한 자료형
# <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

################################ 1번 ####################################
sess = tf.compat.v1.Session() 
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(W)
print(aaa)
sess.close()
# [2.2086694] W의 랜덤 값

################################ 2번 ####################################

sess = tf.compat.v1.InteractiveSession() 
sess.run(tf.compat.v1.global_variables_initializer())
bbb = W.eval() # 변수.eval
print(bbb)
sess.close()
# [2.2086694]

################################ 3번 ####################################

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = W.eval(session=sess)
print(ccc)
sess.close()
# [2.2086694]
#########################################################################

# 다 동일한 값 나온다. 원하는 코드 사용하기.
