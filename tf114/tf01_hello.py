import tensorflow as tf
# print(tf.__version__)
print("hello world")

hello = tf.constant("hello world")   # 파이썬에서 constant는 상수라는 의미

sess = tf.compat.v1.Session()
print(sess.run(hello))






