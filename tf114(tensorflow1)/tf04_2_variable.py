import tensorflow as tf

sess = tf.Session()

x = tf.Variable([2],dtype=tf.float32, name='test')

init = tf.global_variables_initializer()
# 텐서플로우에 쓸 수 있게 모든 변수를 초기화시켜 준다. 문법이니까 그냥 외우기

sess.run(init)

print(sess.run(x))
# [2.]

# WARNING:tensorflow:From c:\Study\tf114(tensorflow1)\tf04_2_variable.py:3: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.
# 원래 1.13버전에서 쓰던 방법이라 워닝뜬다