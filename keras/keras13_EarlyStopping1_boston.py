import numpy as np
import pandas as pd
import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston   # sklearn은 학습 예제가 많음


#1. 데이터
datasets = load_boston()   
x = datasets.data    # 데이터가
y = datasets.target  # y에 들어간다.



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=12345678
                                                    )



#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))



#3. 컴파일. 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1,
                              restore_best_weights=True) 

# 모니터를 하다가 var_loss를 중심으로 중지를 시키겠다.
# val_loss가 10번까지  최소값이 올라가면 중지를 하겠다.  (최소값 이후로 최소값보다 넘어간 횟수가 10번 넘어가면 정지)
# restore_best_weights=True 를 True로하면 최소값이 10번 올라가기 전의 값을 사용한다는 뜻이다.
# mode='min'를 오토로 두면 상황에 맞춰서 최대값, 최소값으로 상황에 맞게 사용해준다.
  # r2 같은 경우는 값이 올라가기 때문에 최대값으로 사용해야한다.
  
start_time = time.time()

hist = model.fit(x_train, y_train, epochs=10000, batch_size=100, verbose=1, validation_split=0.2,
                 callbacks=[earlyStopping])  
"""
Epoch 00033: early stopping 이라는 문구가 뜨면서 정지함.
"""  
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("==================================")
print(hist)    
print("==================================")
print(hist.history)   #history안에 값을 알려줌 loss값 출력해줌  
                      # 딕셔너리안에 리스트 loss : [리스트]
                      # "loss와 val" 두 개의 딕셔너리가 있다.
print("==================================")
print(hist.history['loss']) # 키 벨류에서 로스는 문자이므로 문자로 출력해야한다.
print("==================================")
print(hist.history['val_loss']) # 키 벨류에서 로스는 문자이므로 문자로 출력해야한다.


print("걸린시간 : ", end_time)


y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   # 0 ~ 1까지 지표를 나타낸다 (1에 가까우면 가장 좋다 (1에 마출 것.))
print('r2스코어 : ', r2)


import matplotlib.pyplot as plt
plt.figure(figsize = (9,6))     # figsize(가로길이,세로길이)
plt.plot(hist.history['loss'], marker='.', color='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', color='blue', label='val_loss')
plt.grid()                      # plt.grid(True)
plt.title('asaql')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')   # 그래프가 없는쪽에 알아서 해준다 굳이 명시할 필요 없음
# plt.legend()
plt.show()



