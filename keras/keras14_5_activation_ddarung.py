import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 

#1. 데이터
path = './_data/ddarung/'  
                            
                            
train_set = pd.read_csv(path + 'train.csv', index_col=0)  
 # (1459, 10)


test_set = pd.read_csv(path + 'test.csv', index_col=0) 
test_set = test_set.fillna(method='backfill')    

submission = pd.read_csv(path + 'submission.csv')
submission_set = pd.read_csv('C:/study/study-home/_data/ddarung/submission.csv', index_col=0)

print(train_set.isnull().sum()) # null의 합을 구하겠다.
train_set = train_set.dropna()  # nall이 있는 행 부분을 전체 삭제 


x = train_set.drop(['count'], axis=1)  # drop: 빼버린다. count 제거
                                        
y = train_set['count']  # x에 대한 해답지인 count


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100  
                                                    )


#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=9))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
    
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True) 
                              
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=50,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time()


#4. 결과, 예측
y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test) 

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)

r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)
print("걸린시간 : ", end_time)


# loss :  2733.824462890625
# RMSE :  52.28598795063367
# r2스코어 :  0.6286176263134143
# 걸린시간 :  1656750160.9653602





# y_summit = model.predict(test_set)
      
# submissin_set = pd.read_csv(path + 'submission.csv', index_col=0)
#    # print(submissin.shape)   # (715, 1)
# submissin_set['count'] = y_summit
# submissin_set.to_csv('./_data/ddarung/submission.csv', index = True)
      
