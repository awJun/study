import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
# node3 = node1 + node2
node3 = tf.add(node1, node2)


# print(node3)   
# sess.run을 안하면 이렇게만 나옴 Tensor("add:0", shape=(), dtype=float32) 결과 안나옴 

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(node3))














