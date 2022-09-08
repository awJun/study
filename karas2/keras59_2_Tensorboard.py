import numpy as np
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout, GlobalAveragePooling2D
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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard # 작업한 내용들을 웹상으로 보여주게 해주는 것이다.
es = EarlyStopping(monitor="var_loss", patience=15, mode='auto', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=7, mode='auto', verbose=1, factor=0.5)

tb = TensorBoard(log_dir="D:/study_data/tensorboard_log/_graph", histogram_freq=0, write_graph=True, write_images=True)  # write_graph : 표를 그릴것이냐?  True 네!  / write=images=True 이미지를 그릴거냐   네
# dir: 디렉토리 / 한국말로는 폴더라고 한다. / 즉, 로그가 들어갈 폴더를 지정해라
# D:\study_data\tensorboard_log 폴더를 만듬

# 실행방법 : tensorboard --logdir=. (경로)
# http: //localhost:6006
# http: //127.0.0.1:6006


from keras.optimizers import Adam
learning_rate=0.01
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=5, validation_split=0.4, callbacks=[es, reduce_lr, tb], batch_size=32, verbose=1)   # callbacks=[es, reduce_lr, tb] 훈련시키면 tf는 경로고 fit 안에 항목인 train과 validation 폴더가 해당 경로에 생성된다.
end = time.time()                                                                                                              # 훈련에 관한 데이터가 폴더에 계속 들어가므로 폴더의 용량은 1epoch 돌때마다 커진다.

loss, acc = model.evaluate(x_test, y_test)
print("learning_rate : ",learning_rate)
print("loss : ", round(loss, 4))
print("acc : ", round(acc, 4))  # 소수점 4번째 자리까지 출력하겠다.
print("걸린시간 : ", round(end, 4))

######################[ 시각화 ]############################
import matplotlib.pylab as plt

#1
plt.subplot(2, 1, 1)
plt.plot(hist.history["loss"], marker='.', c='red', label='loss')
plt.plot(hist.history["val_loss"], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title("loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(loc='upper right')

#2
plt.subplot(2, 1, 2)
plt.plot(hist.history["acc"], marker='.', c='red', label='acc')
plt.plot(hist.history["val_acc"], marker='.', c='blue', label='val_acc')
plt.grid()
plt.title("acc")
plt.ylabel("acc")
plt.xlabel("epochs")
plt.legend(["acc", "val_acc"])

plt.show()






















