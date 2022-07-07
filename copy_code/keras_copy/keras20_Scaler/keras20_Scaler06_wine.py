import numpy as np
import time

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_wine

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#.1 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target


# print(x.shape)         # (178, 13)
# print(y.shape)         # (178,)
# print(np.unique(y, return_counts=True))    
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64

from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
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
model.add(Dense(100, activation='relu', input_dim=13))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax')) 

#3. 컴파일. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy']) 

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1,
                              restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=100,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time() -start_time


#4. 평가, 예측

################################################################################
loss, acc = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력
                                            # loss = loss / acc = metrics에서 나온 accuracy 값
print('loss : ', loss)
print('acc : ', acc)


result = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
# print(y_predict)

y_test = np.argmax(y_test, axis=1)
# print(y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)

print("걸린시간 : ", end_time)
#########################################################
"""   
scaler 사용 안함

loss :  4.269172668457031
accuracy :  0.5555555555555556
걸린시간 :  1.297640323638916
"""
#########################################################
"""
scaler = StandardScaler()

loss :  0.15922825038433075
accuracy :  1.0
걸린시간 :  1.3088958263397217
"""
#########################################################
"""
scaler =  MinMaxScaler()

loss :  0.5718794465065002
accuracy :  0.9444444444444444
걸린시간 :  1.2869224548339844
"""
#########################################################
"""
scaler = MaxAbsScaler()

loss :  0.8627053499221802
accuracy :  0.8333333333333334
걸린시간 :  1.328953742980957
"""
#########################################################
"""
scaler = RobustScaler()

loss :  0.19383378326892853
accuracy :  0.9722222222222222
걸린시간 :  1.3113396167755127
"""  
#########################################################
 












