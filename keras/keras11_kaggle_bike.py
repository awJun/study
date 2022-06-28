# 캐글 바이크 문제풀이
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 


#1. 데이터
path = './_data/kaggle_bike/'  
  
     # 결측치는 없는 것으로 판명                                                               
train_set = pd.read_csv(path + 'train.csv', index_col=0)  
  # (10886, 11)                
test_set = pd.read_csv(path + 'test.csv', index_col=0) 
  # (6493, 8)




# axis 관련 링크 : https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=nilsine11202&logNo=221420394534

x = train_set.drop(['casual',  'registered',  'count'], axis=1)  #       axis = 0/1(축값, 0=행, 1=열)                                                                    
   # (10886, 8)
y = train_set['count']
   # (10886,)
   



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.95,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation = 'swish', input_dim=8))
model.add(Dense(100, activation = 'swish'))
model.add(Dense(100, activation = 'swish'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=30)

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
  
      
submissin_set = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
   # print(submissin_set.shape)   # (6493, 1)

submissin_set['count'] = y_summit
submissin_set['count'] = list(map(abs,submissin_set['count']))
submissin_set.to_csv('./_data/kaggle_bike/sampleSubmission.csv', index = True)
      

