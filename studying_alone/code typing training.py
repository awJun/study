
import numpy as np
import pandas as pd
from sqlalchemy import true #pandas : 엑셀땡겨올때 씀
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from keras.layers.recurrent import LSTM, SimpleRNN
import datetime as dt


#1. 데이터
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv', index_col=0) # + 명령어는 문자를 앞문자와 더해줌  index_col=n n번째 컬럼을 인덱스로 인식
            
test_set = pd.read_csv(path + 'test.csv', index_col=0) # 예측에서 쓸거임  3

# 수치형 변수와 범주형 변수 찾기
numerical_feats = train_set.dtypes[train_set.dtypes != "object"].index
categorical_feats = train_set.dtypes[train_set.dtypes == "object"].index
numerical_feats_ = test_set.dtypes[test_set.dtypes != "object"].index
categorical_feats_ = test_set.dtypes[test_set.dtypes == "object"].index
# print("Number of Numberical features: ", len(numerical_feats)) # 37
# print("Number of Categorical features: ", len(categorical_feats)) # 43


train_set_encoded = train_set.drop(numerical_feats,axis=1)
# print(train_set_encoded)

test_set_encoded = test_set.drop(numerical_feats_,axis=1)
# print(test_set_encoded)

le = LabelEncoder()

train_set_encoded.loc[:,:] = \
train_set_encoded.loc[:,:].apply(LabelEncoder().fit_transform)    

# print(train_set_encoded)

train_set = pd.concat([train_set_encoded, train_set.loc[:,numerical_feats]], axis=1)

# print(train_set)


#2. 모델구성
model = Sequential()
model.add(Dense(1, activation = 'swish', input_dim=8))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=300)

#4. 결과, 예측
loss = model.evaluate(x_test, y_test)
print('lose : ', loss)

y_predict = model.predict(x_test)
def RMSE(a, b):
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


#####################################################


y_summit = model.predict(test_set)
  
      
submissin_set = pd.read_csv(path + 'sample_submission.csv', index_col=0)
   # print(submissin_set.shape)   # (6493, 1)

submissin_set['count'] = y_summit
submissin_set['count'] = list(map(abs,submissin_set['count']))
submissin_set.to_csv('./_data/kaggle_house/sample_submission.csv', index = True)














































