import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)   # Variable([2] 나는 변수를 2로 줄거야 ~   / 그리고 공간은 float32로 확보할거야
y = tf.Variable([3], dtype=tf.float32)   # Variable([3] 나는 변수를 3로 줄거야 ~   / 그리고 공간은 float32로 확보할거야

# init = tf.compat.v1.global_variables_initializer()  #전에 줫던 모든 변수들에 대헤서 초기화가 된다 ! 즉 초기값을 넣어줄 수 있는 상태가 된다.
# sess.run(init)   # 초기화 한 후 sess.run을 통화시켜줘야한다.

print(sess.run(x+y))




















