from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time 
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc  

import tensorflow as tf 
tf.random.set_seed(7)  

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)

# loss의 스케일 조정을 위해 0 ~ 255 -> 0 ~ 1 범위로 만들어줌
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

# mean = np.mean(x_train, axis=(0 , 1 , 2 , 3))
# std = np.std(x_train, axis=(0 , 1 , 2 , 3))
# x_train = (x_train-mean)/std
# x_test = (x_test-mean)/std

# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32, 32, 3)
# print(x_train.shape)    # (50000, 32, 32, 3)
# print(np.unique(x_train, return_counts=True))

# One Hot Encoding
from tensorflow.python.keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape)

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), # padding='same', 
                 activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(2, 2))  
model.add(Dropout(0.2))     
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))                
model.add(MaxPooling2D(2, 2))  
model.add(Dropout(0.2))     
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(2, 2))     
model.add(Dropout(0.2))     
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))   
model.add(MaxPooling2D(2, 2))     
model.add(Dropout(0.2))                    
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))  
# model.add(Dropout(0.2))   
             
model.add(Flatten())    
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
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

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min',
                              restore_best_weights=True,
                              verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '02_', date, '_', filename])
                      )
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=128,
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


#그래프로 비교
font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
# plt.title('loss & val_loss')    
plt.title('로스값과 검증로스값')    
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()   
plt.show()









#==[ 정리 ]=======================================================

# astype('float32')
# https://www.educba.com/pandas-dataframe-dot-astype/

#    int형
#   [[179 177 173]
#    [164 164 162]
#    [163 163 161]]

# x_train = x_train.astype('float32') 를 거쳐서 정수 -> 실수로 변경됨

#    float형
#    [ 59.  62.  63.]
#    [ 43.  46.  45.]
#    [ 50.  48.  43.]
   
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
   
# MaxPooling2D
# https://computersciencewiki.org/index.php/File:MaxpoolSample2.png

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# x_train = x_train/255  # 값이 너무 커지기 때문에 이미지 범위가 0 ~ 255이니까 255를 
#                        # train에서 나눠줌 그냥 그렇다고함 (성능은 확실히 좋아짐!)
                         # x의 유니크값이 부동소수형태로 된다. 이경우 컴퓨터에서 성능 좋아짐
#  전 [[179. 177. 173.]
#     [164. 164. 162.]
#     [163. 163. 161.]]


# 후 [[0.23137255 0.24313726 0.24705882]
#     [0.16862746 0.18039216 0.1764706 ] 
#     [0.19607843 0.1882353  0.16862746] 
#    ...
#     [0.61960787 0.5176471  0.42352942] 
#     [0.59607846 0.49019608 0.4       ] 
#     [0.5803922  0.4862745  0.40392157]]

# 컴퓨터는 int보다 float에서 성능이 더 잘나온다.

#--[ 이해말고 걍 쓰자 ㅋ ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# mean = np.mean(x_train, axis=(0 , 1 , 2 , 3)) # MinMaxScaler 역할
# std = np.std(x_train, axis=(0 , 1 , 2 , 3))   # standardScaler 역할
# x_train = (x_train-mean)/std                
# x_test = (x_test-mean)/std

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# reshape

# reshape 괄호안에 원하는 형태의 데이터 숫자를 넣으면 그 형태로 바꿔줌
  # 해당 데이터에서는 같은 형태로 그냥 찍어본거임 무시할것 ㅋ
  # [참고] 50000데이터(열 데이터)를 제외하고 나머지 뒤에 값은
    # 차원의 형태 상관없이 뒤에 값을 모두 곱했을 때 같은 값만 나오면 괜찮음 
    # 2차원으로 바꾸고 싶으면 (50000, 32, 32, 3)를 (50000, 3072)로 하면됨
    
# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(10000, 32, 32, 3)







