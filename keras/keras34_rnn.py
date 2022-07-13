import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN

#1. 데이터
datasets = np.array([1, 2, 3, 4 ,5 ,6 ,7 ,8, 9, 10])

# 하나의 데이터에서 x와 y를 추출함 
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]])
y = np.array([4, 5, 6, 7, 8, 9, 10])


# (n,3) -> (n, 3, 1) 같으나  n, 3를 연산하고 연산한 것을 1개씩 모아서 확인하겠다. 라는 뜻 / 데이터양은 같으므로 같은 형식이다 단시 확인여부차이 rnn은 확인포함해서 항상 3차원이다 ~
#[중요!]# x의_shape = (행, 열, 몇 개씩 자르는지!!!)

print(x.shape)
print(y.shape)

x = x.reshape(7, 3, 1)     # 2차원 데이터를 3차원으로 만들 기 위해 뒤에 1을 곱하는 형태로 만들어줌 (1을 곱하면 데이터값 변동이 없기 때문)

                                                # SimpleRNN를 사용하려면 input이 무조건 3차원이어야 한다.
#2. 모델구성 
model = Sequential()
model.add(SimpleRNN(100,activation='relu' ,input_shape=(3, 1)))   #SimpleRNN를 거치면 3차원이 2차원으로 간다.
# model.add(SimpleRNN(10))       # <-- 3차원이 아니라 2차원을 넣어서 에러가 발생함.[참고]
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = np.array([8, 9, 10]).reshape(1, 3, 1)      #8, 9, 10을 넣어서 11일을 예측       # [중요]rnn 모델에서 사용할 것이므로 3차원으로 변환작업
                                                    # .reshape 앞에 array([8, 9, 10])를 (1, 3, 1)로 바꾸겟다. [[[8], [9], [10]]]

# y_pred안에 np.array([8, 9, 10]) 배열이 3개의 값이 들어 있으므로 

# .reshape(1, 3, 1) 안에 1, 3, 1인 이유는 x.reshape(7, 3, 1)에서 3, 1 부분을  input_shape=(3, 1)에 넣어서 사용해서 3, 1 부분을
  # 넣고 뒤에 1을 곱하는 형식으로 3차원으로 만들어 줬다
result = model.predict(y_pred) 
print("loss : ", loss)
print("[8,9,10의 결과", result)


# loss :  1.1080705064614449e-07
# [8,9,10의 결과 [[11.0072975]]

# loss :  1.8820875311575946e-06
# [8,9,10의 결과 [[11.009936]]







# result = model.predict([8, 9, 10]) #8, 9, 10을 넣어서 11일을 예측  / 이건 여기서 사용 불가 


























































