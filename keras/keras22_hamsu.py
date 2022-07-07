import numpy as np

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
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

# model  = Sequential()
# # model.add(Dense(10, input_dim=3))      
# model.add(Dense(10, input_shape=(3,)))   

# model.add(Dense(5, activation = 'relu'))
# model.add(Dense(3, activation = 'sigmoid'))
# model.add(Dense(1))
##################################################################

### 새로운 모델 ###
input1 = Input(shape=(3,))   # 처음에 Input 명시하고 Input 대한 shape 명시해준다.
dense1 = Dense(10)(input1)   # Dense 구성을하고  node 값을 넣고 받아오고 싶은 변수 받아온다.
dense2 = Dense(5, activation = 'relu')(dense1)    # 받아온 변수를 통해 훈련의 순서를 사용자가 원하는대로 할 수 있다.
dense3 = Dense(3, activation = 'sigmoid')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1) # 해당 모델의 input과 output을 설정한다.

##################################################################


"""   model.summary()


Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 3)]               0
_________________________________________________________________
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

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs=10, batch_size=1)

#4. 평가, 예측















