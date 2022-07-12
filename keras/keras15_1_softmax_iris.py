import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
tf.random.set_seed(66)  # 텐서플로우의 난수를 66으로 넣어서 사용하겠다. weight의 난수
                        # 텐서플로우의 데이터의 난수
                        
                        
#1. 데이터
datasets = load_iris()
print(datasets.DESCR)  #행(Instances): 150   /   열(Attributes): 4
print(datasets.feature_names)

x = datasets['data']  # .data와 동일 
y = datasets['target']  
print(x.shape)   # (150, 4)
print(y.shape)   # (150,)
print("y의 라벨값 : ", np.unique(y))  # 해당 데이터의 고유값을 출력해준다.

from tensorflow.keras.utils import to_categorical   # python까지 넣으면 오류남

# print(y.shape)  (150, 3)
y = to_categorical(y)
# to_categorical를 사용하면 y의 라벨값의 갯수에 맞춰서 알아서 백터의 양을
# 만들어준다.

# print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )
print(y_train)
print(y_test)



#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=4))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 다중분류에선 마지막에 softmax를 사용
# softmax를 사용하면 3개가 출력된다. 3개중 큰쪽으로 찾는다.
# 다중분류 일 때는 최종 노드의 갯수는 y의 라벨의 갯 수 


#3. 컴파일. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy']) 
# categorical_crossentropy를 사용하면 훈련을 할 때 값을 0 과 1로 구분해서 fit을 진행한다.
# categorical_crossentropy는 다중분류에서 사용된다.
# 에러는 멈춘다.   /  버그는 잘돌아가나 값이 이상히게 나온다.
# https://cheris8.github.io/artificial%20intelligence/DL-Keras-Loss-Function/  [loss 관련 링크]

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True) 
  
  
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=100,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time()


#4. 평가, 예측

################################################################################
loss, acc = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력
                                            # loss = loss / acc = metrics에서 나온 accuracy 값
print('loss : ', loss)
print('acc : ', acc)


result = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
print(y_predict)

y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)



   #      수치확인용으로 사용한 것임
# y_predict = model.predict(x_test)
# print("=============================================")
# print(y_predict)
# print("=============================================")
# y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)
# print("=============================================")
# y_predict = to_categorical(y_predict)
# print(y_test)
# print("=============================================")
# print(y_predict)
# print("=============================================")

# # 위에 각 3줄은 같은 모델이다.
# #################################################################################
# print("==============================================")
# print(y_test[:5])
# print("==============================================")
# y_pred = y_predict = model.predict(x_test[:5])
# print(y_pred)
# print("==============================================")



"""
[[0. 0. 1.]                                   <- 다 더한값이 1
 [1. 0. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]]
==============================================
[[2.2621309e-10 9.9886947e-06 9.9998999e-01]   <- 이중에서 제일 큰 놈을 1로 만들고 버린다.
 [9.9999976e-01 2.3192474e-07 3.2410743e-19]
 [5.2614618e-13 1.2018796e-07 9.9999988e-01]
 [1.0000000e+00 6.3018448e-09 1.6175473e-23]
 [2.5301384e-08 1.0616867e-03 9.9893826e-01]]
 
 
"""






# print("걸린시간 : ", end_time)






# y_predict = y_predict.round(0)

# from sklearn.metrics import r2_score, accuracy_score
# acc = accuracy_score(y_test, y_predict)
# print('accuracy : ', acc)

# import matplotlib.pyplot as plt
# plt.figure(figsize = (9,6))     # figsize(가로길이,세로길이)
# plt.plot(hist.history['loss'], marker='.', color='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', color='blue', label='val_loss')
# plt.grid(True, axis=('x'))                      # plt.grid(True)
# plt.title('load_iris')
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend(loc = 'upper right')   # 그래프가 없는쪽에 알아서 해준다 굳이 명시할 필요 없음
# # plt.legend()
# plt.show()



# # loss가 -가 뜨면 지표가 잘못되었다는 뜻


# model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 100)               500
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               10100
# _________________________________________________________________
# dense_2 (Dense)              (None, 100)               10100
# _________________________________________________________________
# dense_3 (Dense)              (None, 100)               10100
# _________________________________________________________________
# dense_4 (Dense)              (None, 3)                 303
# =================================================================
# Total params: 31,103
# Trainable params: 31,103
# Non-trainable params: 0
# _________________________________________________________________













































































