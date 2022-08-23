import tensorflow as tf
tf.set_random_seed(777)
import matplotlib.pyplot as plt

# x_trian = [1, 2, 3]
# y_train = [1, 2, 3]
x_trian = [1]
y_train = [1]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

# w = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name="weight")
w = tf.compat.v1.Variable(tf.compat.v1.Variable(10, dtype=tf.float32)) # 계산해보려고 10으로 해봤음

### [ loss 계산 ] ############################################################################

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))

##### [ 계산 ] ################################################################################
lr = 0.1
gradient = tf.reduce_mean((w * x - y) * x)   # GradientDescentOptimizer를 풀어쓸 것이다. 
descent = w - lr * gradient
update = w.assign(descent)   # assign : 할당하다      

w_history = []
loss_history = []


sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(w))

for step in range(21):
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x:x_trian, y:y_train})
    
    print(step, '\t', loss_v, "\t", w_v)
    
    w_history.append(w_v)
    loss_history.append(loss_v)
    
sess.close()

print("========== W history ===================")
print(w_history)
print("========== loss history ================")
print(loss_history)







