import tensorflow as tf
node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

# 실습
# 덧셈   node3
# 뻴셈   node4
# 곱셈   node5
# 나눗셈 node6

node3 = node1 + node2
sess = tf.compat.v1.Session()

print(sess.run(node3))   # 5.0

 
 
 
# node3 = tf.add_n(node1, node2)


# node5 = tf.matmul(node1, node2)

# node6 = tf.mod(node1, node2)

# sess = tf.compat.v1.Session()

# # 이 세개는 행렬 연산에서 사용
# # TypeError: Tensor objects are only iterable when eager execution is enabled. To iterate over this tensor use tf.map_fn.
# # 이러한 에러가 발생한다.

# # print(sess.run(node3))   # 5.0
# # print(sess.run(node5))   # 6.0
# # print(sess.run(node6))   # 0.6666667


# # tf.add      # 덧셈
# # tf.add_n    # 덧셈

# # tf.subtract # 뺄셈

# # tf.multiply # 곱셈
# # tf.matmul   # 곱셈

# # tf.divide   # 나눗셈
# # tf.mod      # 나눗셈










