import tensorflow as tf
tf.compat.v1.set_random_seed(123)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

y = y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=123,
                                                    stratify=y
                                                    )

print(type(x_train), type(y_train))   # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(x_train.dtype, y_train.dtype)   # 자료형 확인 : dtype      /   float64 int32

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.float32, shape=[30, 1], name="weights")
b = tf.compat.v1.Variable(tf.float32, shape=[1], name="bias")



hypothesis = tf.compat.v1.sigmoid(tf.matmul(x, w) + b) 

#3-1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y_data)) 
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))  # binary_crossentropy 풀어쓴 것
# model.compile(loss='binary_crossentropy')


# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  # 멀티에서는 이렇게 안하니까 nan으로 출력되었음..
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-6)

train = optimizer.minimize(loss)

# 3-2 훈련

with  tf.compat.v1.Session() as sess:   # 나는 tf.compat.v1.Session()를 sess라고 할거야 그리고 같이(with) 실행할거야 / with문을 사용하면 안에서 작업이 끝난 후 자동으로 close를 해준다.
    sess.run(tf.global_variables_initializer())

    epoch = 100
    for epochs in range(epoch):
        # sess.run(train)
        cost_val, hy_val, _ = sess.run([loss, hypothesis, train],    # _의 뜻은 반환하지 않지만 실행은 시키겠다. 라는 뜻
                                   feed_dict={x : x_train, y : y_train})
        # cost_val : loss와 같음   /  hy_val : hypothesis의 값
            
        if epochs % 20 == 0:
            print(epochs, "loss : ", cost_val, "\n", hy_val)    

# # [ accuracy_score ]#########################################################

    y_predict = sess.run(tf.cast(hy_val > 0.5, dtype=tf.float32))   # cast로 반올림 대신 조건걸음 이거말고 반올림도 상관없음
                                                                # cast의 역할은 해당 조건이 만족하면 1 아니면 0으로 반환한다.
    from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
    # acc = accuracy_score(y_data, y_predict)
    acc = tf.reduce_mean(tf.cast(tf.equal(y, y_predict), dtype=tf.float32))
    # print("acc: ", acc)

    # mae = mean_absolute_error(y_data, hy_val)
    # print("mae : ", mae)
    pred, acc = sess.run([y_predict, acc], feed_dict={x:x_test, y:y_test})
    print("===============================================================================")
    print("예측값 : \n", hy_val)
    print("예측결과 : ", pred)
    print("accuracy : ", acc)

