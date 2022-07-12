"""=[ input_shape 사용 ]==============================================================================================

model  = Sequential()
# model.add(Dense(10, input_dim=3))      
# # model.summary()에서 (100, 3)  --> (None, 3)로 받아온다

- - - - - - - - - - - - - - - - 

model  = Sequential()
model.add(Dense(10, input_shape=(3,)))   
# 두 개는 같은 의미로 사용된다 / 차이점은 input_dim은 1차원 개념만 input_shape은 그 이상을 할 때
  사용된다.  input_shape를 사용 할 때 행은 무시하고 열만 괄호안에 넣어주면 된다.

--[ input_shape ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

model  = Sequential()
model.add(Dense(10, input_shape=(3,))) 
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

========================================================================================================================
"""  


import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
             )    # (2,10)  원래는 10행 2열의 데이터를 사용해야하지만 2행 10열인 데이터이므로
                  # 트랜스포스 하여서 사용

# 대활호 하나 = 리스트  

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])  # (10, )

# 계산을 할 때 행렬 연산으로하므로  x, y의 리스트 모양을 같게 해야한다. 

# y=w(1)x + b(10) x 1번째 데이터와 y비교

print(x.shape)  # (3, 10)
print(y.shape)  # (10, )

# x = x.T             # 행과 열을 바꾼다
x = x.transpose()   # 행과 열을 바꾼다.
# x = x.reshape(10,2)   # 순서 유지
print(x)     
print(x.shape)  #(10, 3)


#2. 모델구성
model  = Sequential()
###################################################################
# model.add(Dense(10, input_dim=3))      
# # model.summary()에서 (100, 3)  --> (None, 3)로 받아온다
model.add(Dense(10, input_shape=(3,)))   
# 두 개는 같은 의미로 사용된다 / 차이점은 input_dim은 1차원 개념만 input_shape은 그 이상을 할 때
# 사용된다.  input_shape를 사용 할 때 행은 무시하고 열만 괄호안에 넣어주면 된다.
###################################################################
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))


"""   model.summary()

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 10)                40
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 4
=================================================================
Total params: 117
Trainable params: 117
Non-trainable params: 0
_________________________________________________________________
"""
