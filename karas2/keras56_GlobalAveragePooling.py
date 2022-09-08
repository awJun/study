"""
[해당 프로젝트 설명]
https://colab.research.google.com/drive/1mhdPS1xPpYIDZhe5ZxWXFRV3Zp7RxEab#scrollTo=zi2kmbqqooZc


"""


import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout, GlobalAveragePooling2D
import keras
import tensorflow as tf 


# 1.데이터
(x_train,y_train),(x_test,y_test)  = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# 2.모델 
activation="relu"
drop=0.2
optimizer = "adam"

inputs = Input(shape=(28, 28, 1), name="input")
x = Conv2D(64, (2, 2), padding="valid",
           activation=activation, name='hidden1')(inputs)    # 27, 27, 128  / 27 * 27 * 128
x = Dropout(drop)(x)
# x = Conv2D(64, (2, 2), padding="same",
#            activation=activation, name='hidden2')(x)         # 13, 13, 64
# x = Dropout(drop)(x)
x = Conv2D(32, (3, 3), padding="valid",
           activation=activation, name='hidden3')(x)         # 12, 12, 32
x = Dropout(drop)(x)
# x = Flatten()(x)                                             # None * 25 * 25 * 32  =  20000
x = GlobalAveragePooling2D()(x)

x = Dense(100, activation=activation, name='hidden4')(x)        
x = Dropout(drop)(x)

outputs = Dense(10, activation="softmax", name="outputs")(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()


#3. 컴파일, 훈련
model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')


import time
start = time.time()
model.fit(x_train, y_train, epochs=1, validation_split=0.4,
          batch_size = 128)
end = time.time()


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print("loss :",  loss)
print('acc : ', acc)

# print('걸린시간 : ' , end - start)

print(x_test.shape)
# print(y_predict[:10])
y_predict = np.argmax(model.predict(x_test), axis=-1)
print(y_predict.shape)
print('걸린시간 : ' , end - start)
print('acc : ', accuracy_score(y_test, y_predict))
# print(y_predict)
# print(y_test)
print(y_test)
























