import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout #(데이터의 노드를 중간중간 날려줌)
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
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 1.0000000000000002
# print(np.min(x_test))   # -0.06141956477526944
# print(np.max(x_test))   # 1.1478180091225068


#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=30))
model.add(Dropout(0.3))      # 30%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))       # 20%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

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
"""
Dropout 사용안함

loss :  0.1095195785164833
accuracy :  0.9532163742690059
걸린시간 :  1657103141.0889308
"""
#########################################################
"""
Dropout 사용

loss :  0.10632817447185516
accuracy :  0.9532163742690059
걸린시간 :  1657245835.6482365
"""  
#########################################################
 