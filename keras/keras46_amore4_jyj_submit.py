# 시험~

from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM  
from tensorflow.python.keras.models import Sequential, Input, Model, load_model 
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
삼성_데이터 = pd.read_csv(path + '삼성전자220718.csv', thousands=",", encoding='cp949') # index_col=n n번째 컬럼을 인덱스로 인식

아모레_데이터 = pd.read_csv(path + '아모레220718.csv', thousands=",", encoding='cp949')


def split_x(seq, size): #함수 split_x는 한 
    aaa = []
    for i in range(len(seq) - 24):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)
size=20
samx = split_x(삼성_데이터,size)
amorex = split_x(아모레_데이터,size)
# amorex = split_x(amorex[],size)

size=3
def split_y(seq, size): #함수 split_x는 한 
    aaa = []
    for i in range(len(seq) - 23):
        subset = seq[i+20:(i+20+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)
samy = split_y(삼성_데이터,size)
amorey = split_y(아모레_데이터,size)

#==[아모레 데이터 작업]==========================================================================================================================

# print(아모레_데이터.describe())
# print(아모레_데이터.isnull().sum())  



#==[아모레 데이터 날짜 컬럼 분리 작업]==========================================================================================================================

아모레_데이터['일자'] = pd.to_datetime(아모레_데이터['일자'])
아모레_데이터['연도'] = 아모레_데이터['일자'].dt.year
아모레_데이터['월'] = 아모레_데이터['일자'].dt.month
아모레_데이터['일'] = 아모레_데이터['일자'].dt.day
# print(아모레_데이터.columns)



#==[아모레 데이터 컬럼 정리 작업]==========================================================================================================================
   # 사용할 데이터만 남기자 ~  / 나는 전일비, 일자, 'Unnamed: 6'라는 데이터 즉! 수치 데이터가 아닌 또는 사용하기 싫은 녀석은 안쓸거야 ~

아모레_데이터 = 아모레_데이터.drop(['일자', '전일비', 'Unnamed: 6', '프로그램', '외인비', '외인(수량)', '신용비', '등락률', '금액(백만)'], axis=1)
# print(아모레_데이터.columns)
# Index(['시가', '고가', '저가', '종가', '거래량', '개인', '기관', '외국계', '연도', '월', '일'], dtype='object')



#==[아모레_데이터의 거래량 컬럼 안에 0인 값을 nan으로 대체 작업]===============================================================================================

아모레_데이터['거래량'] = 아모레_데이터['거래량'].replace(0, np.nan)     # 0인 값을 nan으로 처리

# 각 column에 0 갯수 확인
for col in 아모레_데이터.columns:
    missing_rows = 아모레_데이터.loc[아모레_데이터[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))

아모레_데이터 = 아모레_데이터.dropna()
# print(아모레_데이터.isnull().sum())  

아모레_데이터 = 아모레_데이터[:1773]
# print(아모레_데이터.shape) # (1773, 11)

#--[예측 데이터 빼두기 ~]==========================================================================================================

# (예측_데이터)
아모레_예측_시가 = 아모레_데이터['종가']
# print(아모레_예측_시가.shape)  # (1773,)
# 예측_종가 = 아모레_데이터['종가']

# 아모레_예측_시가 = 예측_시가[1]


#==[아모레 데이터 정규화 작업]==========================================================================================================================

from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler = MinMaxScaler()

# 정규화 대상 column 정의
scale_cols = ['고가', '저가', '시가', '거래량', '개인', '기관', '외국계', '연도', '월', '일']

아모레_scale_df = scaler.fit_transform(아모레_데이터[scale_cols])  # <-- np로 변환된 거 같음.. 아마 ? 그래서 오류났음 ;ㅠ 아래에서 pd로 변환
# print(아모레_scale_df)



#==[feature colume(x데이터) / label column(y데이터) 정의 np를 pd로 변환 작업]==========================================================================================================================

아모레_scale_df = pd.DataFrame(아모레_scale_df, columns=scale_cols)
# print(아모레_scale_df)

# {보류 항목 꺼냄 !}-----------------------------------------------------------------------------------------------

# (입력_데이터)
feature_cols = ['고가', '저가', '시가', '거래량', '개인', '기관', '외국계', '연도', '월', '일']
label_아모레 = 아모레_데이터['종가']




feature_아모레 = pd.DataFrame(아모레_scale_df, columns=feature_cols)





#--[DataFrame을 numpy로 변환 작업]- - - - - - - - - - - - - - - - - - - - - -


feature_아모레 = feature_아모레.to_numpy()
label_아모레 = label_아모레.to_numpy()






#==[삼성_데이터 작업]==========================================================================================================================

# print(삼성_데이터.describe())
# print(삼성_데이터.isnull().sum())  



#==[삼성_데이터 날짜 컬럼 분리 작업]==========================================================================================================================

삼성_데이터['일자'] = pd.to_datetime(삼성_데이터['일자'])
삼성_데이터['연도'] = 삼성_데이터['일자'].dt.year
삼성_데이터['월'] = 삼성_데이터['일자'].dt.month
삼성_데이터['일'] = 삼성_데이터['일자'].dt.day
# print(아모레_데이터.columns)



#==[삼성_데이터 컬럼 정리 작업]==========================================================================================================================
   # 사용할 데이터만 남기자 ~  / 나는 전일비, 일자, 'Unnamed: 6'라는 데이터 즉! 수치 데이터가 아닌 또는 사용하기 싫은 녀석은 안쓸거야 ~

삼성_데이터 = 삼성_데이터.drop(['일자', '전일비', 'Unnamed: 6', '프로그램', '외인비', '외인(수량)', '신용비', '등락률', '금액(백만)'], axis=1)
# print(아모레_데이터.columns)
# Index(['시가', '고가', '저가', '종가', '거래량', '개인', '기관', '외국계', '연도', '월', '일'], dtype='object')



#==[삼성_데이터의 거래량 컬럼 안에 0인 값을 nan으로 대체 작업]===============================================================================================

삼성_데이터['거래량'] = 삼성_데이터['거래량'].replace(0, np.nan)     # 0인 값을 nan으로 처리

# 각 column에 0 갯수 확인
for col in 삼성_데이터.columns:
    missing_rows = 삼성_데이터.loc[삼성_데이터[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))

삼성_데이터 = 삼성_데이터.dropna()
# print(아모레_데이터.isnull().sum())  

삼성_데이터 = 삼성_데이터[:1773]
# print(아모레_데이터.shape)  # (1773, 11)



#==[삼성_데이터 정규화 작업]==========================================================================================================================

from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler = MinMaxScaler()

# 정규화 대상 column 정의
scale_cols = ['고가', '저가', '시가', '거래량', '개인', '기관', '외국계', '연도', '월', '일']

삼성_scale_df = scaler.fit_transform(삼성_데이터[scale_cols])  # <-- np로 변환된 거 같음.. 아마 ? 그래서 오류났음 ;ㅠ 아래에서 pd로 변환
# print(아모레_scale_df)


#==[feature colume(x데이터) / label column(y데이터) 정의 np를 pd로 변환 작업]==========================================================================================================================

삼성_scale_df = pd.DataFrame(삼성_scale_df, columns=scale_cols)
# print(아모레_scale_df)

# {보류 항목 꺼냄 !}-----------------------------------------------------------------------------------------------

# (입력_데이터)
feature_cols = ['고가', '저가', '시가', '거래량', '개인', '기관', '외국계', '연도', '월', '일']
label_삼성 = 삼성_데이터['종가']  
# label_cols = ['종가']   # 내일 사용


feature_삼성 = pd.DataFrame(삼성_scale_df, columns=feature_cols)


 
#--[DataFrame을 numpy로 변환 작업]- - - - - - - - - - - - - - - - - - - - - -


feature_삼성 = feature_삼성.to_numpy()
label_삼성 = label_삼성.to_numpy()

# print(label_삼성.shape) # (1773,)
# print(label_아모레.shape) # (1773,)




#================================================================================================================


아모레_x_train, 아모레_x_test, 아모레_y_train, 아모레_y_test, 삼성_x_train, 삼성_x_test, 삼성_y_train, 삼성_y_test = train_test_split(feature_아모레, label_아모레, feature_삼성, label_삼성,
                                                    train_size=0.8,
                                                    shuffle=False,
                                                    random_state=100
                                                    )





#====[ 3차원으로 변환 ]===============================================================

아모레_x_train = 아모레_x_train.reshape(1418, 10, 1)              
아모레_x_test = 아모레_x_test.reshape(355, 10, 1)
아모레_y_train = 아모레_y_train.reshape(1418, 1)
아모레_y_test = 아모레_y_test.reshape(355, 1)

삼성_x_train = 삼성_x_train.reshape(1418, 10, 1)  
삼성_x_test = 삼성_x_test.reshape(355, 10, 1)
삼성_y_train = 삼성_y_train.reshape(1418, 1)
삼성_y_test = 삼성_y_test.reshape(355, 1)



model = load_model('./_test/keras24_ModelCheckPoint.hdf5')




#4. 평가, 예측
y1_loss = model.evaluate([아모레_x_test, 삼성_x_test], [아모레_y_test, 삼성_y_test])  # https://ebbnflow.tistory.com/133
print("아모레, 삼성의 로스값 :  ",y1_loss)

y_predict1, y_predict2 = model.predict([아모레_x_test, 삼성_x_test])


print(y_predict1[-1])


