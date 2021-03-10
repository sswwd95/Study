import tensorflow as tf

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = W * x + b

# print('hypothesis : ', ???)
# 실습 tf07파일 hypothesis 출력되게 만들기
# 1. sess.run()
# 2. InteractiveSession
# 3. .eval(session=sess)

################################ 1번 ####################################

sess = tf.compat.v1.Session() 
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(hypothesis)
print(aaa)
sess.close()
# [1.3       1.6       1.9000001]
################################ 2번 ####################################

sess = tf.compat.v1.InteractiveSession() 
sess.run(tf.compat.v1.global_variables_initializer())
bbb = hypothesis.eval() # 변수.eval
print(bbb)
sess.close()
# [1.3       1.6       1.9000001]

################################ 3번 ####################################

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print(ccc)
sess.close()
# [1.3       1.6       1.9000001]
#########################################################################


# 동재코드
# 1번
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('sess.run() : ', sess.run(hypothesis))

# 2번 
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('sess.run() : ', hypothesis.eval(session = sess))
    
# 3번
with tf.compat.v1.InteractiveSession().as_default() as sess:
    sess.run(tf.global_variables_initializer())
    print('eval() : ', hypothesis.eval())