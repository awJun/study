"""
[해당 프로젝트 설명]
https://colab.research.google.com/drive/18l-xGirdaMAcGBej_fAw9pRs71TqL70g#scrollTo=Ig475zNMwVAA

LR : learning_rate

핵심은 learning_rate를 감축임 왜냐 경사하강법에서 가장 아래에 위치하기 위함 이다
그래서 해당 위치에 얼리스타핑을 걸어서 해야함

얼리스타핑은 필수는 아니지만 통상적으로 얼리스타핑과 많이 사용한다.

"""


import numpy as np
from tensorflow.keras.datasets import mnist, cifar100
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout, GlobalAveragePooling2D
import keras
import tensorflow as tf 


# 1.데이터
(x_train,y_train),(x_test,y_test)  = cifar100.load_data()

x_train = x_train.reshape(50000, 32*32*3)   # .astype('float32')/255.   아래에서 스캐일러햇으므로 주석해줫음
x_test = x_test.reshape(10000, 32*32*3)     # .astype('float32')/255.

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2.모델 
activation="relu"
drop=0.2
optimizer = "adam"

inputs = Input(shape=(32, 32, 3), name="input")
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

outputs = Dense(100, activation="softmax", name="outputs")(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()


#3. 컴파일, 훈련
model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')


import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau # ReduceLROnPlateau : learning_rate를 감축해줌
es = EarlyStopping(monitor="val_loss", patience=20, mode="min", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=10, model="auto", verbose=1, factor=0.5) # factor=0.5 : 50%만큼 learning_rate를 감축해줄거야
                                                                                                    # 0.001 (50%감축)-> 0.0005 (50%감축)-> 0.00025  즉, 조건이 거릴ㄹ 때 마다 50%씩 감축해준다.
                                                                                                    # 이렇게 흘러가다가 얼리스타핑 걸리면 바로 훈련 정지시키는 원리임!
start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.4,
          batch_size=128, callbacks=[es, reduce_lr])
end = time.time()


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print("loss :",  loss)
print('acc : ', acc)

print('걸린시간 : ' , end - start)

# print(y_predict[:10])
# y_predict = np.argmax(model.predict(x_test), axis=-1)

# print('걸린시간 : ' , end - start)
# print('acc : ', accuracy_score(y_test, y_predict))









































