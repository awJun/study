# placeholder는 입력에서 밖에 못쓴다. 즉, 애는 데이터를 정의하는 용도라고함(즉!, 넣어주는 역할밖에 못함)  /  애는 변하지 않는 수   
# variable(변수)  /  constant(상수)
import tensorflow as tf
import numpy as np
print(tf.__version__)
print(tf.executing_eagerly())  # True 사용불가라는 뜻

# 즉시실행모드 
tf.compat.v1.disable_eager_execution()  # 꺼 !
print(tf.executing_eagerly())  # False 사용가능이라는 뜻

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

################### 요기서부터 #####################

sess = tf.compat.v1.Session()
a = tf.compat.v1.placeholder(tf.float32)   # 각각의 공간을 만듬
b = tf.compat.v1.placeholder(tf.float32)   # 각각의 공간을 만듬

add_node = a + b
print(sess.run(add_node, feed_dict={a:3, b:4.5}))   # placeholder를 정의하고 feed_dict에서 각각을 정의해줘야 한다.   / 7.5
print(sess.run(add_node, feed_dict={a:[1, 3], b:[2, 4]}))    #[3. 7.]

add_and_triple = add_node * 3
print(add_and_triple)      # Tensor("mul:0", dtype=float32)

print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))   # 22.5   

# 그래프를 만들고 sess.run을 통해서 그 그래프를 우리가 볼 수 있게 하는 것이다.
















