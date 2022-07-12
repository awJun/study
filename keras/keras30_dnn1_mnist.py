from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D ,Dropout   
from tensorflow.python.keras.models import Sequential
from keras.datasets import mnist,cifar10,cifar100,fashion_mnist
import pandas as pd
import numpy as np

import tensorflow as tf            
tf.random.set_seed(66)  # 모델 구성에서 처음 w에 들어가는 값은 원래 랜덤으로 들어가지만  
                          # 랜덤이 아닌 66번째 난수표에있는 값을 처음 w에 넣어서 사용하겠다.
#성능은 CNN 보다 좋게

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)           # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)             # (10000, 28, 28) (10000,) 
                                              # reshape할때 모든 객체의 곱은 같아야 한다, 순서는 바뀌면 안되지만 모양만 바꾸면 된다

# import matplotlib.pyplot as plt
# plt.imshow(x_train[5], 'gray')    # x_train 데이터중 고유 라벨값 5번의 데이터를 출력해라 (그레이 색인 그림이므로 그레이 적음)
# plt.show()                        # 이 렇게 출력한 데이터 형식이 그림 형식이다


print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),  y의 형태가 2차원인데 x의 형태는 3차원이므로 이것을 2차원으로 바꿔주는 작업
                                              # array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))
# unique는 1차원일 경우 사용 불가능하지만 2차원 이상부터는 사용이 가능하다.

#-----[ x와 y의 데이터 형태를 마춤 x_train데이터(3차원)을 2차원으로 ]

x_train = x_train.reshape(60000, 28*28)  # 기존의 x_train shape를 28*28로 모양을 바꾸겟다 (reshape 이것이)   
                # reshape를 사용하면 행별로 잘라서 1행으로 이어서 붙힘 이경우 우리는 못알아보지만 컴퓨터는 알아볼 수 있기에 사용
x_test = x_test.reshape(10000, 28*28)    # [주의] x.T를하면 "데이터가 일그러져서"(숫자 5형태 데이터가 망가짐) 알아볼 수 없게됨 컴퓨터도 인식 힘듬
print(x_train.shape) # (60000, 784)
print(y_train.shape) # (60000,)
 #  y값의 라벨이 먼지 확인해야한다
                                              
#-----[ x와 y의 데이터 형태를 마춤 x_train데이터(3차원)을 2차원으로 ]

y_train = pd.get_dummies(y_train)   
y_test = pd.get_dummies(y_test)
print(pd.get_dummies(y_train))
print(y_train.shape)

#----[y의 고유의 값이 10개이기 때문에 1개인 열을 10개로 바꿔주기 위해서 get_dummies를 사용함 ]

print(y_train.shape) # (60000, 10)  



# x_train = x_train.reshape(60000, 28, 28, 1)  #----[다시 원래대로 돌리는 방법 계산 후]
                
# x_test = x_test.reshape(10000, 28, 28, 1)



#2. 모델구성
model = Sequential()

# model.add(Dense(64, input_shape=(28*28, ))) - 28x28 사이즈였다는걸 이런식으로 명시도 가능 
model.add(Dense(64, input_shape=(784, )))  # Dense를 사용할 경우 2차원으로 변경을해야 사용이 가능
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint      
earlyStopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1,
                              restore_best_weights=True)        


hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2,
                 callbacks=[earlyStopping], verbose=1)  
                                            # batch_size:32 디폴트값 3번정도 말 한듯


#4. 결과, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0]) 
print('accuracy : ', results[1]) 


from sklearn.metrics import r2_score, accuracy_score 
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
print(y_predict)

y_test = tf.argmax(y_test, axis=1)
print(y_test)


acc = accuracy_score(y_test, y_predict)  
print('acc스코어 :', acc)

















