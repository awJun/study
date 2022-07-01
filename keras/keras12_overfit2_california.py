import numpy as np
import time

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()

x=datasets.data
y=datasets.target




x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100
                                                    )



#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일. 훈련
model.compile(loss='mae', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=30000, batch_size=3000, validation_split=0.25)

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("걸린시간 : ", end_time)


import matplotlib.pyplot as plt
plt.figure(figsize = (9,6))     # figsize(가로길이,세로길이)
plt.plot(hist.history['loss'], marker='.', color='red', label='loss')
# label='loss' =  plt.legend에 출력되는 라벨 이름
# marker='.' 표에서 각 구간을 .으로 찍어서 보여줌 굳이 사용안해도 상관없음.
plt.plot(hist.history['val_loss'], marker='.', color='blue', label='val_loss')
# plt.legend에 출력되는 라벨 이름

plt.grid()   # plt.grid(True) axis에 x, y외에 넣으면 오류발생,  디폴트 값 = True
plt.title('asaql')              # 그래프 제목 
plt.ylabel('loss')              # y축 이름
plt.xlabel('epochs')            # x축 이름
plt.legend(loc = 'upper right') # 그래프가 없는쪽에 알아서 해준다 굳이 명시할 필요 없음
# plt.legend()
plt.show()





























