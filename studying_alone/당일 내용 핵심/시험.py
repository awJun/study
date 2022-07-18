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
삼성_데이터 = pd.read_csv(path + '삼성전자_데이터.csv', thousands=",", encoding='cp949', index_col=0,) # index_col=n n번째 컬럼을 인덱스로 인식

아모레_데이터 = pd.read_csv(path + '아모레_데이터.csv', thousands=",", encoding='cp949', index_col=0)



#==[아모레 데이터 작업]==========================================================================================================================

# print(아모레_데이터.describe())
# print(아모레_데이터.isnull().sum())  


# 아모레_데이터의 거래량 컬럼 안에 0인 값을 nan으로 대체 작업
아모레_데이터['거래량'] = 아모레_데이터['거래량'].replace(0, np.nan)     # 0인 값을 nan으로 처리

# 각 column에 0 갯수 확인
for col in 아모레_데이터.columns:
    missing_rows = 아모레_데이터.loc[아모레_데이터[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))

아모레_데이터 = 아모레_데이터.dropna()
# print(아모레_데이터.isnull().sum())  








#==[아모레 데이터 컬럼 정리 작업]==========================================================================================================================
   # 사용할 데이터만 남기자 ~  / 나는 전일비라는 화살표 데이터 즉! 수치 데이터가 아닌 녀석을 안쓸거야 ~

# 아모레_데이터 = 아모레_데이터[["시가", "고가", "저가", "종가", ]]



#==[아모레 데이터 정규화 작업]==========================================================================================================================

from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler = MinMaxScaler()

# 정규화 대상 column 정의
scale_cols = []











# #=[삼성_데이터 작업]==========================
# #최대한 많은 양 잘라주기
# 삼성_데이터 = 삼성_데이터.loc[:3041]
# 아모레_데이터 = 아모레_데이터.loc[:3181]
# print(삼성_데이터.loc[0],삼성_데이터.loc[0])
# # print(삼성_데이터.shape, 삼성_데이터.shape)
# #dropcol 정하기
# # print(삼성_데이터.columns)
# #모델구성위해 x,y로 나누기
# # sx = d_s.

# #=============================================

























































































