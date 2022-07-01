# 만들어서 깃허브에 올릴 것
# 발리 성능 비교

import numpy as np
import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

datasets = load_diabetes()

x = datasets.data
y = datasets.target

'''
print(x)
print(y) # y는 어차피 비교 대상일 뿐이므로 전처리가 안되어도 상관없다.

print(x.shape, y.shape)  # (442, 10) (442,)  # 백터 1개

print(datasets.feature_names)
print(datasets.DESCR)
'''

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100
                                                    )



#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일. 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=500, batch_size=10,  validation_split=0.25)

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

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































