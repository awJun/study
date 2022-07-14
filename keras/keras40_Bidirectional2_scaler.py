import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.layers import Bidirectional  # Bidirectional는 python안에 들어가지 못해서 계속 오류가 발생하는 거 같음 걍 일단 이렇게 쓰자 ;  
                                                    # 위에도 다 똑같이 python을 빼줘야함 ..ㅠ

import numpy as np
import time
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. 데이터
datasets = np.array([1, 2, 3, 4 ,5 ,6 ,7 ,8, 9, 10])

# 하나의 데이터에서 x와 y를 추출함 
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]])
y = np.array([4, 5, 6, 7, 8, 9, 10])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

#--[해당 데이터의 unique형태 확인]- - - - - - - - - - - - - - - - - 
# print(np.unique(x_train, return_counts=True))
# print(np.unique(y_train, return_counts=True))    # <--- 이 항목은 확인 필수   회귀모델이다. 여기서는 데이터가 너무 작아서 착각하기 쉬웠던 거 같음 .ㅠ
# print(np.unique(x_test, return_counts=True))   
# print(np.unique(y_test, return_counts=True))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#--[해당 데이터 shape 확인]- - - - - - - - - - - - - - - - - - - - -
print(x_train.shape)   # (5, 3)
print(y_train.shape)   # (5,)  
print(x_test.shape)    # (2, 3)
print(y_test.shape)    # (2,) 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
#--[스케일러를 사용하기 위해 차원 변형 작업]- - ( 데이터의 형태가 2x2가 아닐때만 사용할 것 )- - - -
#  ( 아래 스케일러들는 2x2형태에서만 돌아가기 때문에  (60000, 28, 28)형태를 2x2로 변환하는 작업임 )

#[사용 안함]
# x_train = x_train.reshape(50000, 3072)              
# x_test = x_test.reshape(10000, 3072)

# print(x_train.shape)  # (50000, 3072)    
# print(x_test.shape)   # (10000, 3072)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[ 스케일러 작업]- - - - - - - - - - - - - - - - - -(데이터 안에 값들의 차이을 줄여줌(평균으로 만들어주는 작업))
# scaler =  MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
                                
scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[차원 변형 작업]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
x_train = x_train.reshape(5, 3, 1)              
x_test = x_test.reshape(2, 3, 1)

print(x_train.shape)  # (5, 3, 1)
print(x_test.shape)   # (2, 3, 1)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#- -[y데이터를 x와 동일한 차원 형태로 변환] - - - - - - - - - - - - - - - - - - - - ( [중요]!! 회귀형에서는 할 필요없음  )
# from tensorflow.python.keras.utils.np_utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) #(120, 3) (30, 3)
#----------------------------------------------------------------------------------

#==[ Bidirectional을 위에서 ]======================================================================================
# 2. 모델구성 
model = Sequential()
# model.add(SimpleRNN(100,activation='relu' ,input_shape=(3, 1))) 
model.add(Bidirectional(SimpleRNN(100, return_sequences=True, activation='relu') ,input_shape=(3, 1)))  # Bidirectional 뒤쪽에만 성능이 좋고 앞에는 의미가 없을 정도로 성능이 구린 현상을 방지하기 위해
                                                                                                # Bidirectional으로 랩핑(감싸주기)를 시전했음 ~   
                                                                                                # Bidirectional는 무조건 첫번째 아래에서 사용해야하는 듯 함 오류남 ㅋ ;;
                                                                                                # 아니였어 !!! return_sequences=True랑 동시에 해서 그래... 랩핑이라 오류임
                                                                                                # 그러니까 Bidirectional를 사용 후 return_sequences=True를 아래에서 시전하자
model.add(SimpleRNN(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.summary()

# # =================================================================================================================                                                                                     
#==[ Bidirectional을 아래에서 ]======================================================================================
# # 2. 모델구성 
# model = Sequential()
# # model.add(SimpleRNN(100,activation='relu' ,input_shape=(3, 1))) 
# model.add(SimpleRNN(100, activation='relu' ,input_shape=(3, 1), return_sequences=True ))    # Bidirectional 뒤쪽에만 성능이 좋고 앞에는 의미가 없을 정도로 성능이 구린 현상을 방지하기 위해
#                                                                                                 # Bidirectional으로 랩핑(감싸주기)를 시전했음 ~   
# model.add(Bidirectional(SimpleRNN(5)))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1))
# model.summary()

# # =================================================================================================================

# # Bidirectional로 랩핑했을 때 나온 결과임 !
# # Model: "sequential"
# # _________________________________________________________________
# #  Layer (type)                Output Shape              Param #   
# # =================================================================
# #  simple_rnn (SimpleRNN)      (None, 3, 100)            10200     

# #  bidirectional (Bidirectiona  (None, 10)               1060      
# #  l)

# #  dense (Dense)               (None, 100)               1100      

# #  dense_1 (Dense)             (None, 300)               30300     

# #  dense_2 (Dense)             (None, 300)               90300     

# #  dense_3 (Dense)             (None, 300)               90300     

# #  dense_4 (Dense)             (None, 300)               90300     

# #  dense_5 (Dense)             (None, 300)               90300     

# #  dense_6 (Dense)             (None, 100)               30100

# #  dense_7 (Dense)             (None, 1)                 101

# # =================================================================
# # Total params: 434,061
# # Trainable params: 434,061
# # Non-trainable params: 0
# # _________________________________________________________________





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

# loss :  5.278317388203446e-13
# [8,9,10의 결과 [[10.7247505]]


# loss :  4.0977003664011136e-05
# [8,9,10의 결과 [[10.861145]]


