import tensorflow as tf

node1= tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)

print(node3)
# tf.Tensor(7.0, shape=(), dtype=float32) -> 2점대
# Tensor("Add:0", shape=(), dtype=float32) -> 1점대

# 사람이 볼 수 있게 만들려면 sess.run 나와야한다. (2점대에선 sess.run이 사라짐)
sess = tf.Session()
print('sess.run(node3) : ', sess.run(node3))
print('sess.run(node3) : ', sess.run([node1,node2]))
# sess.run(node3) :  7.0
# sess.run(node3) :  [3.0, 4.0]