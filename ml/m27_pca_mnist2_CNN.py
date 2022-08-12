# [실습]
# 아까 4가지 모델을 맹그러봐
# 784개 DNN으로 만든거(최상의 성능인거 // 0.996이상)과 비교!!

# time 체크 

#1. 나의 최고의 DNN
# time = ???
# acc = ???

#2. 나의 최고의 CNN
# time = ???
# acc = ???

#--[ 위와 아래 비교 ]   1번 2번보다 3, 4, 5, 6번이 성능이 더 좋게 나오게 맹그러봐

#3. PCA 0.95
# time = ???
# acc = ???

#4. PCA 0.99
# time = ???
# acc = ???

#5. PCA 0.999
# time = ???
# acc = ???

#6. PCA 1.0
# time = ???
# acc = ???

##[ 시작! ]################################################################################

from keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import numpy as np
x = np.append(x_train, x_test, axis=0)
# print(x.shape) # (70000, 28, 28)

x = x.reshape(70000, 28*28) # (70000, 784)
# print(x.shape)

from sklearn.decomposition import PCA
pca = PCA(n_components=403)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_

cumsum =np.cumsum(pca_EVR)
# print(cumsum)
# print(np.argmax(cumsum >= 0.996) + 1)  

#--[ train과 test 분리 ]---------------------------------------------------------------------

x_train = x[:60000] # (60000, 403)  # 6만번째까지
x_test = x[60000:]  # (10000, 403)  # 6만번째 이후까지

#--[ 차원 변환]---------------------------------------------------------------------

x_train = x_train.reshape(60000, 403, 1, 1)
x_test = x_test.reshape(10000, 403, 1, 1)

#--[ y데이터 교유값에 맞춰서 열 맞추기 ]---------------------------------------------
from tensorflow.python.keras.utils.np_utils import to_categorical # 범주
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten  
model = Sequential() 
model.add(Conv2D(filters=64, kernel_size=(1,1), padding='same', input_shape=(403, 1, 1)))
                                              # padding 0이라는 패딩을 씌워서 이미지를 조각낼때 가장자리 부분을 두번 이상 넣어줘서 다른 부분보다 덜 학습되는걸 방지 
                                              # 통상 shape를 다음 레이어에도 유지하고 싶을때 padding을 쓴다                                                                                    
model.add(Conv2D(32, (1, 1), padding='valid', activation='relu'))    # padding='valid' 디폴트
model.add(Conv2D(32, (1, 1), activation='relu'))    
model.add(Conv2D(32, (1, 1), activation='relu'))   
model.add(Flatten())                          # (N, 252)   Flatten을 안써도 하단 dense로 계산된다                                                             
model.add(Dense(32, activation='relu'))    
model.add(Dropout(0.2))                                                                                                     
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1,
                              restore_best_weights=True)        

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1, batch_size=100, validation_split=0.2,
                 callbacks=[earlyStopping], verbose=1)  
end_time = time.time() -start_time

#4. 결과, 예측
import tensorflow as tf   

results = model.evaluate(x_test, y_test)
# print('loss : ', results[0]) 
# print('accuracy : ', results[1]) 


from sklearn.metrics import accuracy_score 
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
# print(y_predict)

y_test = tf.argmax(y_test, axis=1)
# print(y_test)


acc = accuracy_score(y_test, y_predict)  
print('acc스코어 :', acc)
print('time :', end_time)


# acc스코어 : 0.9143
# time : 105.35770726203918


# 아래 에러 해결은 stratify = y를 사용하기
# ValueError: Invalid classes inferred from unupue values of 'y'. Expected:
#     [0 1 2 3 4 5 6], got [1 2 3 4 5 6 7]


# xgboost가 느려서 gpu를 사용하려고 할 때 
# model. XGBClassifier(여기에다가)
# tree_method='gpu_hist', predictor='gpu_predictor', gpu_id = 0
# 를 넣으면 된다.


