import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
import tensorflow as tf

from sqlalchemy import false

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.datasets import fetch_covtype

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#1. 데이터

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y)) # [1 2 3 4 5 6 7]
# print(x,y)

# print(datasets.DESCR)
# print(datasets.feature_names)
# print(datasets)

# print(x)
# print(y)
####################케라스########################
# y = to_categorical(y)
# print(np.unique(y, return_counts=True)) # y의 라벨값 :  [1 2 3 4 5 6 7]
#################################################

####################겟더미#######################
# y = pd.get_dummies(y)
# print(y)
################################################

####################원핫인코더###################
df = pd.DataFrame(y)
print(df)
oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
y = oh.fit_transform(df)
print(y)
################################################


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

# print(y_test)
# print(y_train)
# print(y)


# scaler =  MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 1.0000000000000002
# print(np.min(x_test))   # -0.06141956477526944
# print(np.max(x_test))   # 1.1478180091225068


#2. 모델
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout #(데이터의 노드를 중간중간 날려줌)


### 기존 모델 ###

model = Sequential()
model.add(Dense(500, input_dim=54, activation='linear')) #sigmoid : 이진분류일때 아웃풋에 activation = 'sigmoid' 라고 넣어줘서 아웃풋 값 범위를 0에서 1로 제한해줌
model.add(Dropout(0.3))      # 30%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))       # 20%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(16, activation='relu'))             
model.add(Dense(7, activation='softmax'))             # softmax : 다중분류일때 아웃풋에 활성화함수로 넣어줌, 아웃풋에서 소프트맥스 활성화 함수를 씌워 주면 그 합은 무조건 1로 변함
                                                                 # ex 70, 20, 10 -> 0.7, 0.2, 0.1


#3. 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=1000, mode='auto', verbose=1, 
                              restore_best_weights=True)   
start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=100,
                 validation_split=0.2,
                 callbacks=[es],
                 verbose=1)
end_time = time.time() -start_time

#4. 평가, 예측
loss, acc= model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', acc)

# results= model.evaluate(x_test, y_test)
# print('loss : ', results[0])
# print('accuracy : ', results[1])


y_predict = model.predict(x_test)
print(y_predict)
print(y_test)
y_predict = np.argmax(y_predict, axis= 1)  # 판다스 겟더미 쓸때는 tf.argmax sklearn 원핫인코딩 쓸때는 np
print(y_predict)
y_test = np.argmax(y_test, axis= 1)
print(y_test)
# y_predict = to_categorical(y_predict)
# y_test = np.argmax(y_test, axis= 1)
print(np.unique(y_predict))
print(np.unique(y_test))



acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc) 

print(" end_time : ", end_time)



#########################################################
"""
Dropout 사용안함


"""
#########################################################
"""
Dropout 사용

loss :  0.12056204676628113
accuracy :  0.975
걸린시간 :  7.065335750579834
"""  
#########################################################
 