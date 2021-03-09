import tensorflow as tf
######## 상수 고정의 문제가 있다 ######
node1= tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)
#####################################

sess = tf.Session()

# placeholder는 input의 개념(sess.run을 통과시키기 전에 넣고 싶은 값을 넣을 수 있다)
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5})) # adder_node : 더하기
# 7.5
# placeholder는 인풋을 할 것을 지정을 해주고 sess.run할 때 feed_dict에서 지정해준다.

print(sess.run(adder_node, feed_dict={a:[1,3], b:[3,4]}))
# [4. 7.]
# feed_dict={a:3, b:4.5} 딕셔너리형식 -> 리스트 넣을 수 있다

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:4, b:2}))
# 4와 2를 더한거에 3을 곱한다
# 18.0

