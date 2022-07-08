import numpy as np
import time

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout #(데이터의 노드를 중간중간 날려줌)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import load_digits

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


#.1 데이터
datasets = load_digits()
x = datasets.data     # (1797, 64)
y = datasets.target   # (1797,)


from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
# print(y)
# print("+++++++++++++++++++++++++")
# print(x)

# print(x.shape) # (1797, 64)
# print(y.shape)   # (1797, 10)



# x: 17  /  y: 2
# print("y의 라벨값 : ", np.unique(x))


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
model.add(Dense(100, activation='relu', input_dim=64))
model.add(Dropout(0.3))      # 30%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))       # 20%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax')) 

#3. 컴파일. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy']) 

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=100,
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

y_test = np.argmax(y_test, axis=1)   # 해당 리스트에서 값이 큰 스칼라의 인덱스 번호를 출력 
# print(y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)

print("걸린시간 : ", end_time)



#########################################################
"""
Dropout 사용안함

loss :  0.13947711884975433
accuracy :  0.9666666666666667
걸린시간 :  9.515319108963013

"""
#########################################################
"""
Dropout 사용

loss :  0.12056204676628113
accuracy :  0.975
걸린시간 :  7.065335750579834
"""  
#########################################################
 




