from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, MaxPooling1D, Conv1D
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

import tensorflow as tf
tf.random.set_seed(777)

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

# loss의 스케일 조정을 위해 0 ~ 255 -> 0 ~ 1 범위로 만들어줌
x_train = x_train.astype('float32')  # astype() 메서드는 계열의 값을 int 유형에서 float 유형으로 변환하는 데 사용됩니다
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

mean = np.mean(x_train, axis=(0 , 1 , 2 , 3))
std = np.std(x_train, axis=(0 , 1 , 2 , 3))
x_train = (x_train-mean)/std
x_test = (x_test-mean)/std

x_train = x_train.reshape(50000, 96, 32)
x_test = x_test.reshape(10000, 96, 32)
print(x_train.shape)    # (50000, 32, 32, 3)
print(np.unique(x_train, return_counts=True))

# [ One Hot Encoding ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
from tensorflow.python.keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)

# 원핫인코은 y에서 분류할때만 사용된다. x에다가 넣으면 안됨 !
# x는 훈련 데이터인데 원핫을 때리면 원핫의 가장큰 문제중 하나인 값이 들어있는 부분을 제외하고 모든 부분은 다 0으로
# 채운다는 문제점이 있다. 그렇기 때문에 x에다가는 원핫은 안된다고함 ! 아무튼 쓰지마 ~~~
  # 내 생각엔 y를 고유값에 맞춰서 아웃풋에 사용해야 하므로 그런 듯 함 ?  회귀형은 몰르겟다 ㅋ 회기랑 이중분류 아웃풋은 걍 1개 ㅋ
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#2. 모델링 
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=(3), # padding='same', 
                 activation='relu', input_shape=(96, 32)))
          # filters <= 노드갯수
model.add(MaxPooling1D(2, 2))  
model.add(Dropout(0.2))     
model.add(Conv1D(32, (3), padding='same', activation='relu'))                
model.add(MaxPooling1D(2))  
model.add(Dropout(0.2))     
model.add(Conv1D(64, (3), padding='same', activation='relu'))
model.add(MaxPooling1D(2))     
model.add(Dropout(0.2))     
model.add(Conv1D(64, (3), padding='same', activation='relu'))   
model.add(MaxPooling1D(2))     
model.add(Dropout(0.2))                    
model.add(Conv1D(128, (3), padding='same', activation='relu'))  
# model.add(Dropout(0.2))   
             
model.add(Flatten())    
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='Nadam', 
              metrics=['accuracy'])

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
print(date)

filepath = './_ModelCheckPoint/k28/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '02_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=128,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

y_predict = model.predict(x_test)
y_predict = y_predict.argmax(axis=1)      
y_test = y_test.argmax(axis=1)            

acc = accuracy_score(y_test, y_predict)
print("=====================================================================")   
print('acc 스코어 : ', acc)  

print("=====================================================================")
print("걸린시간 : ", end_time)


# loss :  1.3394992351531982
# accuracy :  0.525600016117096
# 걸린시간 :  82.08771872520447























