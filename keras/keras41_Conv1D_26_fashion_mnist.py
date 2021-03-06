import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import numpy as np
import time

import tensorflow as tf
tf.random.set_seed(66)

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28)  # (60000, 28, 28) (60000,)
x_test = x_test.reshape(10000, 28, 28)    # (10000, 28, 28) (10000,)
# print(x_train.shape)    # (60000, 28, 28)
# print(np.unique(x_train, return_counts=True))

import pandas as pd
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train)
print(y_train.shape)


#2. 모델링 
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=(4),    
                 padding='same', input_shape=(28, 28)))
model.add(MaxPooling1D(2))           
model.add(Dropout(0.2))
model.add(Conv1D(64, (4), padding='same', activation='relu'))                
model.add(MaxPooling1D(2))          
model.add(Dropout(0.2))
model.add(Conv1D(64, (4), padding='same', activation='relu'))
model.add(MaxPooling1D(2, 2))          
model.add(Dropout(0.2))

model.add(Flatten())    
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k28/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=30, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '04_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1, batch_size=128,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

import tensorflow as tf
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)   
y_test = tf.argmax(y_test, axis=1)          

acc = accuracy_score(y_test, y_predict)
print("=====================================================================")   
print('acc 스코어 : ', acc)  

print("=====================================================================")
print("걸린시간 : ", end_time)


# loss :  0.8272647261619568
# accuracy :  0.7099000215530396
# 걸린시간 :  3.316495418548584














# #그래프로 비교
# font_path = 'C:\Windows\Fonts\malgun.ttf'
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
# plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
# plt.grid()    
# plt.title('로스값과 검증로스값')    
# plt.ylabel('loss')
# plt.xlabel('epochs')
# plt.legend()   
# plt.show()