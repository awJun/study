import tensorflow as tf
tf.compat.v1.set_random_seed(123)

변수 = tf.compat.v1.Variable(tf.random_normal([1]), name="weiht")   # name="weiht" 변수에 대한 또다른 이름을 줌 그냥 준거임 
print(변수)

#1. 초기화 첫 번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)
print('aaa : ', aaa)   # aaa :  [-1.5080816]
sess.close() 


#2. 초기화 두 번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess)   # <-- session=sess   close하려고 한거임   /  이과정을 거쳐야 진정한 남자(x) 군대....
print("bbb", bbb)                                                          # 이 과정을 거쳐야 진정한 변수로 태어난다......
sess.close()


#3. 초기화 세 번째
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print("ccc : ", ccc)
sess.close()







