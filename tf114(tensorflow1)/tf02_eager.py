# 즉시 실행 모드
# from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf
print(tf.executing_eagerly()) # false

tf.compat.v1.disable_eager_execution() 
# 이걸 사용해서 끄면 2점대에서도 사용 가능, 즉시 실행 모드를 끄는 것
print(tf.executing_eagerly())
# False

# 2점대에서는 
# True
# False

# sessrun이 없어졌다



print(tf.__version__)
# 1.14.0

hello = tf.constant('hello world')
# constant = 상수
print(hello)
# Tensor("Const:0", shape=(), dtype=string) 자료형으로 나온다
# 3가지의 자료형이 존재

# 텐서플로우 1점대에서 자료형이 통과하려면 항상 세션을 만들어야 한다. 
sess = tf.compat.v1.Session()
print(sess.run(hello))
# b'hello world'
# 앞에 b 붙는건.. 신경쓰지마


# AttributeError: module 'tensorflow' has no attribute 'Session' -> 2점대에서는 session이 사라졌다