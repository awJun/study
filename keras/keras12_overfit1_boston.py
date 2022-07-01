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
hist = model.fit(x_train, y_train, epochs=13, batch_size=1, verbose=1, validation_split=0.2)

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



