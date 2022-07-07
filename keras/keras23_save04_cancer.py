import numpy as np
from sklearn.datasets import load_breast_cancer

import time
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR )
# # (569, 30)
print(datasets.feature_names)

x = datasets.data  # [data] = /data 안에 키 벨류가 있으므로 똑같은 형태다.
y = datasets.target 
print(x)
print(y)

# print(x.shape, y.shape) (569, 30) (569,)  y는 569개

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

# scaler =  MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 1.0000000000000002
# print(np.min(x_test))   # -0.06141956477526944
# print(np.max(x_test))   # 1.1478180091225068


#2. 모델구성
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
"""
### 기존 모델 ###

model = Sequential()
model.add(Dense(100, activation='linear', input_dim=30))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='linear'))
model.add(Dense(1, activation='sigmoid'))
"""
### 새로운 모델 ###
input1 = Input(shape=(30,))   # 처음에 Input 명시하고 Input 대한 shape 명시해준다.
dense1 = Dense(100)(input1)   # Dense 구성을하고  node 값을 넣고 받아오고 싶은 변수 받아온다.
dense2 = Dense(100, activation = 'relu')(dense1)    # 받아온 변수를 통해 훈련의 순서를 사용자가 원하는대로 할 수 있다.
dense3 = Dense(100, activation = 'sigmoid')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1) # 해당 모델의 input과 output을 설정한다.


#3. 컴파일. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy', 'mse']) # loss에 accuracy도 같이 보여달라고 선언한 것이다. 로스외에 다른 지표도 같이 출력해준다.
    # 여기서 mse는 회귀모델에서 사용하는 활성화 함수이므로 분류모델에서는 신용성은 없다. 
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True) 
  

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=50,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time()

model.save("./_save/keras23_4_save_cancer.h5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)



y_predict = model.predict(x_test)
y_predict = y_predict.round(0)


from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)

print("걸린시간 : ", end_time)

# # r2 = r2_score(y_test, y_predict)   # 0 ~ 1까지 지표를 나타낸다 (1에 가까우면 가장 좋다 (1에 마출 것.))

#########################################################
"""   [best_scaler]

scaler = StandardScaler()

loss :  [0.09968791157007217, 0.9707602262496948, 0.35551637411117554]
accuracy :  0.6491228070175439
걸린시간 :  1657161247.5155907
"""
#########################################################
"""   
scaler 사용 안함

loss :  [0.2148212343454361, 0.9122806787490845, 0.06270847469568253]       
accuracy :  0.9122807017543859
걸린시간 :  1657097578.4769702  
"""
#########################################################
"""
scaler = StandardScaler()

loss :  [0.0813656821846962, 0.9649122953414917, 0.02246694080531597]       
accuracy :  0.9649122807017544
걸린시간 :  1657097607.6152306  
"""
#########################################################
"""
scaler =  MinMaxScaler()

loss :  [0.07699520140886307, 0.9590643048286438, 0.023455815389752388]     
accuracy :  0.9590643274853801
걸린시간 :  1657097653.847337
"""
#########################################################
"""
scaler = MaxAbsScaler()

[0.1095195785164833, 0.9532163739204407, 0.03458350524306297]
accuracy :  0.9532163742690059
걸린시간 :  1657103141.0889308
"""
#########################################################
"""
scaler = RobustScaler()

[0.07867364585399628, 0.9649122953414917, 0.021984117105603218]
accuracy :  0.9649122807017544
걸린시간 :  1657103102.4072146
"""  
#########################################################
 