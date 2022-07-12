===[ 데이터의 유형 ]==================================================================================================================

# 선형대수에서 다루는 데이터는 개수나 형태에 따라 크게 스칼라(scalar), 벡터(vector), 행렬(matrix), 텐서(tensor) 유형으로 나뉜다
# 스칼라(0차원): 숫자 하나로 이루어진 데이터 x =[1,2,3]에서 1,2,3 하나를 말함 스칼라 3개 
# 벡터(1차원): 스칼라의 모임 x =[1,2,3] 벡터1개 shape=(3,) 
# 행렬(matrix,2차원):벡터,즉 데이터 레코드가 여러인 데이터 집합 ([1,2,3],[4,3,2]) shape =(2,3)
# 텐서(3차원):같은 크기의 행렬이 여러 개 있는 것([[1,2,3],[4,3,2]] , [[4,3,11],[3,7,16]]) shape =(2,2,3) 가장 작은것부터 읽는다 (행,렬)
# tensorflow = 텐서를 연산시키다 피쳐의 숫자는 동일하다.

#  1.행무시 열우선 2.2개이상은 리스트















===[ output이 두 개인 유형 ]==================================================================================================================

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
# print(range(10)) # range: 범위, 거리 0~10까지의 정수형 숫자 

#---[ 알고만 있기 ]------------------------------
# for i in range(10):          # for :반복하라 
#     print(i)
#---------------------------------------------

# print(x) # (3, 10)
# [[  0   1   2   3   4   5   6   7   8   9]
#  [ 21  22  23  24  25  26  27  28  29  30]
#  [201 202 203 204 205 206 207 208 209 210]]

print(x.shape) # (3, 10)

x = x.T # (10, 3)
print(x.shape)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]])


#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=3))   
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(2)) # y의 백터가 두 개이므로 최종 y의 값도 두 개를 출력해야 한다.

















===[ output이 세 개인 유형 ]==================================================================================================================

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])
# print(range(10)) # range: 범위, 거리 0~10까지의 정수형 숫자 

#---[ 알고만 있기 ]------------------------------
# for i in range(10):          # for :반복하라 
#     print(i)
#---------------------------------------------

# print(x) # (3, 10)
# [[  0   1   2   3   4   5   6   7   8   9]
#  [ 21  22  23  24  25  26  27  28  29  30]
#  [201 202 203 204 205 206 207 208 209 210]]

print(x.shape) # (3, 10)

x = np.transpose(x) # (10, 3)
print(x.shape)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
# y의 배열이 3개이므로 3차원 텐서로 분류한다.

print(y.shape)  # (3, 10)

y = np.transpose(y) # (10, 3)
print(y.shape)

#2. 모델
model = Sequential()
model.add(Dense(5, input_dim=3))   
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(3)) # y의 백터가 세 개이므로 최종 y의 값도 두 개를 출력해야 한다.

