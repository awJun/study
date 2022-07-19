# 시험~

from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM   
from tensorflow.python.keras.models import Sequential
import pandas as pd
import numpy as np
import tensorflow as tf       
import time
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
#삼전 10/03/24부터 모레 09/09/01 

#1. 데이터
path = './_data/test_amore_0718/'
삼성_데이터 = pd.read_csv(path + '삼성전자_데이터.csv', thousands=",", encoding='cp949') # index_col=n n번째 컬럼을 인덱스로 인식

아모레_데이터 = pd.read_csv(path + '아모레_데이터.csv', thousands=",", encoding='cp949')


def split_x(a, b):
    임의의_리스트선언 = []
    for i in range(len(a) - b + 1):   # 10 - 5 + 1  =6 /  i가 반복하는 건 6번  /  0 1 2 3 4 5 <--  i가 들어가는 값의 순서
        subset = a[i : (i + b)]    #  subset = dataset[0 : 5]   1, 2, 3, 4, 5 
        임의의_리스트선언.append(subset)   
    
       
    return np.array(임의의_리스트선언)

#--[trainset 만드는 과정]-----------
size = 16   #  x = 4개  y는 1개
train = split_x(아모레_데이터, size)

# print(train)         
# print(train.shape)    # (3165, 16, 17)
# #-----------------------------------


#---[trainset에서 x와 y데이터를 추출하는 과정]---------------------------------------------------------------------------------------------------------
# 1, 2, 3, 4를 x데이터로 만드는 과정 5번째 열은 뺌 / 왜냐하면 "1, 2, 3, 4에 대한 예측은 5"라는 형태의 데이터로 만들기 위해 y에서 사용할 것이기 때문이다.
x = train[:, :-1]     
# # x에서 빼버린 5번째 열을 y데이터로 사용하겟다는 뜻 ~ 
y = train[:, -1]       
#-----------------------------------------------------------------------------------------------------------------------------------------------------
print(x)

# #-[예측 단계중 predict에서 사용 할 데이터 만드는 과정]- - - - - - - - - - -
# size = 4
# test = split_x(testset, size)

# # print(test)
# # print(x.shape, y.shape, test.shape)

# #- - - - - - - - - - - - - - - - -

# # 모델 구성 및 평가 예측할 것.

# x_train, x_test, y_train, y_test = train_test_split(x, y,
#                                                     train_size=0.8,
#                                                     shuffle=True,
#                                                     random_state=100
#                                                     )
























































