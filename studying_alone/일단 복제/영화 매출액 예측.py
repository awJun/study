# 모든 데이터 불러오기

import pandas as pd
path = './review_data/영화 매출 데이터셋/'

한산_용의출현 = pd.read_csv(path + '한산_용의출현.csv', thousands=",", sep='\t')
미니언즈2 = pd.read_csv(path + '미니언즈2.csv', thousands=",", sep='\t')
탑건 = pd.read_csv(path + '탑건.csv', thousands=",", sep='\t')
외계인 = pd.read_csv(path + '외계인.csv', thousands=",", sep='\t')
뽀로로_극장판 = pd.read_csv(path + '뽀로로 극장판 드래곤 캐슬 대모험.csv', thousands=",", sep='\t')
헤어질_결심 = pd.read_csv(path + '헤어질 결심.csv', thousands=",", sep='\t')
명탐정_코난 = pd.read_csv(path + '명탐정 코난 할로윈의 신부.csv', thousands=",", sep='\t')
프리_후편 = pd.read_csv(path + '프리_더 파이널 스트로크 후편.csv', thousands=",", sep='\t')
엑소의_사다리 = pd.read_csv(path + '엑소의 사다리 타고 세계여행3.csv', thousands=",", sep='\t')
토르_러브 = pd.read_csv(path + '토르_러브 앤 썬더.csv', thousands=",", sep='\t')

end_game = pd.read_csv(path + 'end_game.csv', thousands=",", sep='\t')



# y데이터로 사용할 컬럼 따로 저장
y_data = end_game['매출액']
# print(y_data.shape)  # (134,)
end_game = end_game.drop(['매출액'], axis=1)
# print(end_game)
# print(y_data)

# DataFrame을 numpy로 변환 작업
np_x_end_game = end_game.to_numpy()
# print(np_x_end_game)
np_y_end_game = y_data.to_numpy()
# print(np_y_end_game)



from sklearn.model_selection import train_test_split 
x_train_end_game, x_test_end_game, y_train_end_game, y_test_end_game = train_test_split(np_x_end_game, np_y_end_game, 
                                                                                        train_size=0.8,
                                                                                        shuffle=False)
# print(x_train_end_game.shape)   # (107, 6)
# print(x_test_end_game.shape)    # (27, 6)
# print(y_train_end_game.shape)   # (107,)
# print(y_test_end_game.shape)    # (27,)


x_train_end_game = x_train_end_game.reshape(107, 6, 1)              
x_test_end_game = x_test_end_game.reshape(27, 6, 1)
y_train_end_game = y_train_end_game.reshape(107, 1)
y_test_end_game = y_test_end_game.reshape(27, 1)



#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM 
model = Sequential()  # input_shape=(3, 1) == input_length=3, input_dim=1)
                      # [batch, timesteps, feature]
model.add(LSTM(units=64 ,input_length=6, input_dim=1))  
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
# model.summary()




#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['acc'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

import datetime
date = datetime.datetime.now()      
date = date.strftime("%m%d_%H%M")  
# print(date)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
#                       filepath= "".join([filepath, 'k24_', date, '_', filename])
#                       )

es = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1,
                   restore_best_weights=True) 

model.fit(x_train_end_game, y_train_end_game,
          epochs=1000,
          validation_split=0.2,
          batch_size=32,
          verbose=1,
          callbacks=[es])


#4. 평가 및 예측
# loss = model.evaluate(x_test_end_game, y_test_end_game) 

# y_predict = model.predict(y_test_end_game)

# from sklearn.metrics import r2_score
# y1_r2 = r2_score(y_test_end_game, y_predict)   

# # print(y_아모레_데이터[-1:])


# #--[predict 전 reshape 작업]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# print("영화 예측 :  ", loss)
# print("예상 매출가", y1_r2[-1])
# # print('영화 매출가 : ', y1_r2)
# # print('걸린시간 : ', end_time) 






