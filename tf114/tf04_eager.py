import tensorflow as tf
print(tf.executing_eagerly())   # False

# 즉시 실행모드!!   / 2점 버전 / 안키면 1점 버전  텐서2에서 쓸거면 이걸 쓰면 에러 안남 1에서 써도 에러 안남 1이면 위에꺼ㅗ 써도 에러 안남
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())   # False    텐서2에서 사용이 가능하지만 시스템이 충돌해서 에러가 발생할 수 있으므로 가급적으로 1을 사용할 것.

hello = tf.constant("Hello World")

sess = tf.compat.v1.Session()
print(sess.run(hello))              # b'Hello World' /  즉시 실행모드 켠 상태


# 텐서플로2 버전에서는 sess.run을 사용 안해서 텐서2로 돌리면 아래와 같은 에러 발생
# RuntimeError: The Session graph is empty. Add operations to the graph before calling run().

# 즉시 실행모드를
# b'Hello World'

# 즉시 실행모드를 켜면 2점버전 안키면 1점 버전이다.

