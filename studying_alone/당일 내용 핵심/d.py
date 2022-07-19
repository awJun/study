import numpy as np
from sklearn import datasets

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split # 함수 가능성이 높음


trainset = np.array(range(1,101))  # range(1, 11-1)           # print(dataset)  # [ 1  ~ 100 ]
testset = np.array(range(96,106))  # (96 ~ 105)               # 96, 97, 98, 99에대한 예측 이런씩으로 자르기

# print(len(trainset))

def split_x(a, b):
    임의의_리스트선언 = []
    for i in range(len(a) - b + 1):   # 10 - 5 + 1  =6 /  i가 반복하는 건 6번  /  0 1 2 3 4 5 <--  i가 들어가는 값의 순서
        subset = a[i : (i + b)]    #  subset = dataset[0 : 5]   1, 2, 3, 4, 5 
        임의의_리스트선언.append(subset)   
    
       
    return np.array(임의의_리스트선언)

#--[trainset 만드는 과정]-----------
size = 5   #  x = 4개  y는 1개
train = split_x(trainset, size)

# print(train)         
# print(train.shape)    # (96, 5)
#-----------------------------------
print(train)

#---[trainset에서 x와 y데이터를 추출하는 과정]---------------------------------------------------------------------------------------------------------
# 1, 2, 3, 4를 x데이터로 만드는 과정 5번째 열은 뺌 / 왜냐하면 "1, 2, 3, 4에 대한 예측은 5"라는 형태의 데이터로 만들기 위해 y에서 사용할 것이기 때문이다.
x = train[:, :-1]     
# x에서 빼버린 5번째 열을 y데이터로 사용하겟다는 뜻 ~ 
y = train[:, -1]   


# print(x)














