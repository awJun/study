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
                                               #input_dim = feature
x = x.reshape(7, 3, 1)    # [batch, timesteps, feature]
 # 2차원 데이터를 3차원으로 만들 기 위해 뒤에 1을 곱하는 형태로 만들어줌 (1을 곱하면 데이터값 변동이 없기 때문)
 
                                                # SimpleRNN를 사용하려면 input이 무조건 3차원이어야 한다.
#2. 모델구성 
model = Sequential()
model.add(SimpleRNN(units=10,activation='relu' ,input_shape=(3, 1)))   #SimpleRNN를 거치면 3차원이 2차원으로 간다.        #
# model.add(SimpleRNN(10))       # <-- 3차원이 아니라 2차원을 넣어서 에러가 발생함.[참고]
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
model.summary()  # https://velog.io/@yelim421/%EC%88%9C%ED%99%98-%EC%8B%A0%EA%B2%BD%EB%A7%9D-Recurrent-Neural-NetworkRNN / Param가 120인 이유 연산과정


# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# simple_rnn (SimpleRNN)       (None, 10)                120
# _________________________________________________________________
# dense (Dense)                (None, 5)                 55
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 181
# Trainable params: 181
# Non-trainable params: 0
# _________________________________________________________________





# ==> Param 결과해석
# recurrent_weights + input_weights + biases
# (units*units) + (units*features) + units
# units*(units+features) + units
# 10(10+1) + 10 = 120
























































