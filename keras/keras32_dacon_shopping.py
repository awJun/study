# # 데이콘 따릉이 문제풀이
# import numpy as np
# import pandas as pd
#   # pandas = 데이터를 불러올 때 사용 (이외에도 기능도 많다.)

# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error 
#   # mean_squared_error: mse이다. 여기서 루트를 하면 rmse가 된다.

# #1. 데이터
# # "". 현재폴더 / 하단에" 라는 뜻
# path = './_data/ddarung/'   # <- 폴더 경로만 있다. 폴더 안에 어던 데이터셋을 가져올지 선언은 안함
#                             # 그러므로 'train.csv' 를 통해서 데이터를 가져옴
                            
                            
# train_set = pd.read_csv(path + 'train.csv', index_col=0)  
#              # https://mizykk.tistory.com/16
#              # path + 'train.csv   ==   './_data/ddarung/train.csv'
#              # index_col=  첫 번째 열로 정할 인덱스 번호를 입력
             
# print(train_set)

# print(train_set.shape) # (1459, 10) 원래 열이 11개지만, id를 인덱스로 제외하여 10개
#                        # [1459 rows x 10 columns]
#                        # (1459, 10)

# # # test_set는 예측에서 쓸 예정 ~
# test_set = pd.read_csv(path + 'test.csv', index_col=0) 
# print(test_set)
# print(test_set.shape)  # [715 rows x 9 columns]
#                        # (715, 9)
                       
# test_set = test_set.fillna(method='backfill')    


# """
# 대회사이트 링크  https://dacon.io/competitions/open/235576/data


# train.csv 안에서 데이터셋과 테스트셋을 모두 알아서하여 회귀모델을 만들고 test.csv로 y 제출

# 데이터셋 안에 맨뒤에 있는 count은 결과값이므로 제외시키고 사용
# 맨 앞은 index_col으로 앞으로 뺌
# """



# print(test_set.columns)  # 컬럼의 이름을 알려줌
# print(train_set.info())  # non-null -> 데이터가 빠져있다.   null 값이없다          
#                          # 이빨이 빠진 데이터 이것을 결측치 라고한다.

# print(train_set.describe())  # describe: 서술하다 묘사하다.  해당 값 묘사해서 알려줌

# #### 결측치 처리 1. 제거 ####
# print(train_set.isnull().sum()) # null의 합을 구하겠다.
# train_set = train_set.dropna()  # nall이 있는 행 부분을 전체 삭제 
#                                 # 해당 행 부분의 데이터를 다 삭제하므로 데이터 손실이 크다.
# print(train_set.isnull().sum())
# print(train_set.shape)          # (1328, 10)   <- 살아남은 데이터양
# ############################\


#   # x 선언
# x = train_set.drop(['count'], axis=1)   # drop: 빼버린다. count 제거

# print(x)
# print(x.columns)
# print(x.shape)  # (1459, 9) inpip dim = 9  <- 원래 데이터양은 위에 train_set.dropna를 
#                 #                             거쳐서 데이터가 줄었다. (현재는 이 값이 아니다.)
                                               
                                               

#   # y 선언
# y = train_set['count']  # x에 대한 해답지인 count

# print(y)
# print(y.shape)   # (1459,) output = 1개





# x_train, x_test, y_train, y_test = train_test_split(x, y,
#                                                     train_size=0.9,
#                                                     shuffle=True,
#                                                     random_state=100  # 1430
#                                                     )
# # 114     1450(49)  

# '''
# #2. 모델구성
# model = Sequential()
# model.add(Dense(50, input_dim=9))  # 첫번째 히든에 1넣으면 성능 아작난다.
# model.add(Dense(70))
# model.add(Dense(60))
# model.add(Dense(70))
# model.add(Dense(70))
# model.add(Dense(50))
# model.add(Dense(50))
# model.add(Dense(70))
# model.add(Dense(50))
# model.add(Dense(30))
# model.add(Dense(1))
# '''
# #2. 모델구성
# model = Sequential()
# model.add(Dense(1, input_dim=9))  # 첫번째 히든에 1넣으면 성능 아작난다.


# #3. 컴파일, 훈련
# model.compile(loss='mae', optimizer='adam')
# model.fit(x_train, y_train, epochs=1                   
#           , batch_size=3)

# #4. 결과, 예측
# loss = model.evaluate(x_test, y_test)
# print('lose : ', loss)

# y_predict = model.predict(x_test)
# def RMSE(a, b):
#     return np.sqrt(mean_squared_error(a, b))

# rmse = RMSE(y_test, y_predict)
# print("RMSE : ", rmse)


# #####################################################

# y_summit = model.predict(test_set)

# #print(y_summit)
# #print(y_summit.shape)    # (715, 1)

#      ############# .to_csv() 
#      ##### submission.csv를 완성하시오. 

     
      
# submissin_set = pd.read_csv(path + 'submission.csv', index_col=0)
#    # print(submissin.shape)   # (715, 1)
      

# submissin_set['count'] = y_summit


# submissin_set.to_csv('./_data/ddarung/submission.csv', index = True)



#=============================================================================================      

import numpy as np
import pandas as pd
  # pandas = 데이터를 불러올 때 사용 (이외에도 기능도 많다.)
  
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
  # mean_squared_error: mse이다. 여기서 루트를 하면 rmse가 된다.

#1. 데이터
path = './_data/dacon_shopping/'   # <- 폴더 경로만 있다. 폴더 안에 어던 데이터셋을 가져올지 선언은 안함
                            # 그러므로 'train.csv' 를 통해서 데이터를 가져옴

train_set = pd.read_csv(path + 'train.csv', index_col=0)  
             # https://mizykk.tistory.com/16
             # path + 'train.csv   ==   './_data/ddarung/train.csv'
             # index_col=  첫 번째 열로 정할 인덱스 번호를 입력

             
# print(train_set)
#       Store  ... Weekly_Sales
# id           ...
# 1         1  ...   1643690.90
# 2         1  ...   1641957.44
# 3         1  ...   1611968.17
# 4         1  ...   1409727.59
# 5         1  ...   1554806.68
# ...     ...  ...          ...
# 6251     45  ...    734297.87
# 6252     45  ...    766512.66
# 6253     45  ...    702238.27
# 6254     45  ...    723086.20
# 6255     45  ...    713173.95

# [6255 rows x 12 columns]

# print(train_set.shape) #  (6255, 12)


# print(train_set.info())
#  #   Column        Non-Null Count  Dtype        
# ---  ------        --------------  -----        
#  0   Store         6255 non-null   int64        
#  1   Date          6255 non-null   object       
#  2   Temperature   6255 non-null   float64      
#  3   Fuel_Price    6255 non-null   float64      
#  4   Promotion1    2102 non-null   float64      
#  5   Promotion2    1592 non-null   float64      
#  6   Promotion3    1885 non-null   float64      
#  7   Promotion4    1819 non-null   float64      
#  8   Promotion5    2115 non-null   float64      
#  9   Unemployment  6255 non-null   float64      
#  10  IsHoliday     6255 non-null   bool
#  11  Weekly_Sales  6255 non-null   float64      
# dtypes: bool(1), float64(9), int64(1), object(1)memory usage: 592.5+ KB
# None

# print(train_set.isnull().sum()) 
# Store              0
# Date               0
# Temperature        0
# Fuel_Price         0
# Promotion1      4153
# Promotion2      4663
# Promotion3      4370
# Promotion4      4436
# Promotion5      4140
# Unemployment       0
# IsHoliday          0
# Weekly_Sales       0
# dtype: int64

train_set = train_set.fillna(method='backfill')  



# test_set는 예측에서 쓸 예정 ~
test_set = pd.read_csv(path + 'test.csv', index_col=0) 


# print(test_set)
#      Store        Date  ...  Unemployment  IsHoliday
# id                      ...

# 1        1  05/10/2012  ...         6.573      False
# 2        1  12/10/2012  ...         6.573      False
# 3        1  19/10/2012  ...         6.573      False
# 4        1  26/10/2012  ...         6.573      False
# 5        2  05/10/2012  ...         6.170      False
# ..     ...         ...  ...           ...       
#  ...
# 176     44  26/10/2012  ...         5.217      False
# 177     45  05/10/2012  ...         8.667      False
# 178     45  12/10/2012  ...         8.667      False
# 179     45  19/10/2012  ...         8.667      False
# 180     45  26/10/2012  ...         8.667      False

# [180 rows x 11 columns]


# print(test_set.shape)  # (6255, 12)
                       

  
# print(test_set.info())
#  #   Column        Non-Null Count  Dtype        
# ---  ------        --------------  -----        
#  0   Store         180 non-null    int64        
#  1   Date          180 non-null    object       
#  2   Temperature   180 non-null    float64      
#  3   Fuel_Price    180 non-null    float64      
#  4   Promotion1    178 non-null    float64      
#  5   Promotion2    45 non-null     float64      
#  6   Promotion3    161 non-null    float64      
#  7   Promotion4    146 non-null    float64      
#  8   Promotion5    180 non-null    float64      
#  9   Unemployment  180 non-null    float64      
#  10  IsHoliday     180 non-null    bool
# dtypes: bool(1), float64(
    
# print(test_set.isnull().sum()) 
# Store             0
# Date              0
# Temperature       0
# Fuel_Price        0
# Promotion1        2
# Promotion2      135
# Promotion3       19
# Promotion4       34
# Promotion5        0
# Unemployment      0
# IsHoliday         0
# dtype: int64

test_set = test_set.fillna(method='backfill')  


# print(train_set.isnull().sum()) 

# print(test_set.isnull().sum()) 


#   # x 선언
# x = train_set.drop(['count'], axis=1)   # drop: 빼버린다. count 제거

# print(x)
# print(x.columns)
# print(x.shape)  # (1459, 9) inpip dim = 9  <- 원래 데이터양은 위에 train_set.dropna를 
#                 #                             거쳐서 데이터가 줄었다. (현재는 이 값이 아니다.)
                                               
                                               

#   # y 선언
# y = train_set['count']  # x에 대한 해답지인 count

# print(y)
# print(y.shape)   # (1459,) output = 1개


print("")
print(x.columns)























