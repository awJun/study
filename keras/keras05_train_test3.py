import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# [검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법 찾아라

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size = 0.7,  # (둘 중에 하나만 사용하면 된다.)
                                                    test_size=0.3, # (데이터 비율)
                                                    shuffle=True,  # 셔플 데이터를 섞겟다(true면 섞이고 false면 안섞음)
                                                    random_state=12345678) # 랜덤난수 표중에서 66번의 난수값을 사용해라

                                         
print(x_train)  # [2 7 6 3 4 8 5]
print(x_test)   # [ 1  9 10]
print(y_train)  # [2 7 6 3 4 8 5]
print(y_test)   # [ 1  9 10]






#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')      # 70%를 훈련
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)  # x y 대신  % 30를 평가 예측
print('loss : ', loss)

result = model.predict([11])
print('[11]의 예측값 : ', result)

# loss :  2.1626081434078515e-05
# [11]의 예측값 :  [[10.9932995]]























































