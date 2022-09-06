import tensorflow as tf

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w1 = tf.compat.v1.Variable(tf.random.normal([2, 1], name="weights1"))   # 행렬의 곱
b1 = tf.compat.v1.Variable(tf.random_normal([30], name="bias1"))        # 행렬의 합

Hidden_layer = tf.compat.v1.sigmoid(tf.matmul(x, w1) + b1)
# model.add(Dense(30, input_shape=(2,), activation="sigmoid"))

dropout_layers = tf.compat.v1.nn.dropout(Hidden_layer, rate=0.3)   # 히든레이어에 드랍아웃을 쏴버리겟다는 뜻이다.
                                                                        # keep_prob=0.7  :  70%를 살리고 30%를 드랍하겠다.
                                                                        # rate=0.3  :  30를 드랍하겠다. 
                                                                        # 둘 중에 하나를 골라서 사용하면 된다.
print(Hidden_layer)   # Tensor("Sigmoid:0", shape=(?, 30), dtype=float32)
# summy는 모델에서 출력한 것을 한 곳에 모아둔 것이라고함

print(dropout_layers)  # Tensor("dropout/mul_1:0", shape=(?, 30), dtype=float32)














