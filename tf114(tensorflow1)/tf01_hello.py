import tensorflow as tf
print(tf.__version__)
# 1.14.0

hello = tf.constant('hello world')
# constant = 상수
print(hello)
# Tensor("Const:0", shape=(), dtype=string) 자료형으로 나온다
# 3가지의 자료형이 존재

# 텐서플로우 1점대에서 자료형이 통과하려면 항상 세션을 만들어야 한다. 
sess = tf.Session()
print(sess.run(hello))
# b'hello world'
# 앞에 b 붙는건.. 신경쓰지마
