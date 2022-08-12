# 만났던 문제점들 
"""
ValueError: Data cardinality is ambiguous:
  x sizes: 28, 28, 28, 28, 24
  y sizes: 28, 28, 28, 28, 24
Make sure all arrays contain the same number of samples.
앙상블 모델의 행이 달라서 에러가 발생한 것이였음.. ㅠ



"""
### 영화 매출 예측 앙상블 모델 ###

#1_1.[ 데이터 불러오기 ]###################################################################
import pandas as pd
탑건 = pd.read_csv('./review_data/영화 훈련 데이터/탑건.csv', sep='\t') 
아이언맨 = pd.read_csv('./review_data/영화 훈련 데이터/아이언맨.csv', sep='\t') 
어벤져스 = pd.read_csv('./review_data/영화 훈련 데이터/어벤져스.csv', sep='\t') 
엔드게임 = pd.read_csv('./review_data/영화 훈련 데이터/엔드게임.csv', sep='\t') 
닥터스트레인지 = pd.read_csv('./review_data/영화 훈련 데이터/닥터스트레인지.csv', sep='\t') 
토르 = pd.read_csv('./review_data/영화 훈련 데이터/토르.csv', sep='\t') 
# print(아이언맨.shape)         # (118, 7)
# print(어벤져스.shape)         # (118, 7)
# print(엔드게임.shape)         # (118, 7)
# print(탑건.shape)             # (31, 7)
# print(닥터스트레인지.shape)   # (31, 7)
# print(토르.shape)             # (31, 7)

# 1_2.[ 날짜 데이터 분리 ]###################################################################

탑건['일자'] = pd.to_datetime(탑건['날짜'])
탑건['연도'] = 탑건['일자'].dt.year
탑건['월'] = 탑건['일자'].dt.month
탑건['일'] = 탑건['일자'].dt.day

아이언맨['일자'] = pd.to_datetime(아이언맨['날짜'])
아이언맨['연도'] = 아이언맨['일자'].dt.year
아이언맨['월'] = 아이언맨['일자'].dt.month
아이언맨['일'] = 아이언맨['일자'].dt.day

어벤져스['일자'] = pd.to_datetime(어벤져스['날짜'])
어벤져스['연도'] = 어벤져스['일자'].dt.year
어벤져스['월'] = 어벤져스['일자'].dt.month
어벤져스['일'] = 어벤져스['일자'].dt.day

엔드게임['일자'] = pd.to_datetime(엔드게임['날짜'])
엔드게임['연도'] = 엔드게임['일자'].dt.year
엔드게임['월'] = 엔드게임['일자'].dt.month
엔드게임['일'] = 엔드게임['일자'].dt.day

닥터스트레인지['일자'] = pd.to_datetime(닥터스트레인지['날짜'])
닥터스트레인지['연도'] = 닥터스트레인지['일자'].dt.year
닥터스트레인지['월'] = 닥터스트레인지['일자'].dt.month
닥터스트레인지['일'] = 닥터스트레인지['일자'].dt.day

토르['일자'] = pd.to_datetime(토르['날짜'])
토르['연도'] = 토르['일자'].dt.year
토르['월'] = 토르['일자'].dt.month
토르['일'] = 토르['일자'].dt.day


#1_3.[ 데이터 안에 0인 값 확인 ]###################################################################

for col in 탑건.columns:
    missing_rows = 탑건.loc[탑건[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))
for col in 아이언맨.columns:
    missing_rows = 아이언맨.loc[아이언맨[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))
for col in 어벤져스.columns:
    missing_rows = 어벤져스.loc[어벤져스[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))
for col in 엔드게임.columns:
    missing_rows = 엔드게임.loc[엔드게임[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))
for col in 닥터스트레인지.columns:
    missing_rows = 닥터스트레인지.loc[닥터스트레인지[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))
for col in 토르.columns:
    missing_rows = 토르.loc[토르[col]==0].shape[0]
    # print(col + ': ' + str(missing_rows))
# 0인 값은 없는 것으로 판명


#1_4.[ nan 확인 및 제거 ]###################################################################
# print(탑건.isnull().sum())  
# print(아이언맨.isnull().sum())  
# print(어벤져스.isnull().sum())  
# print(엔드게임.isnull().sum())  
# print(닥터스트레인지.isnull().sum())  
# print(토르.isnull().sum())  

# 없는 것으로 판명



#1_5.[ y데이터 분리 ]###################################################################

탑건_매출액 = 탑건['매출액']
아이언맨_매출액 = 아이언맨['매출액']
어벤져스_매출액 = 어벤져스['매출액']
엔드게임_매출액 = 엔드게임['매출액']
닥터스트레인지_매출액 = 닥터스트레인지['매출액']
토르_매출액 = 토르['매출액']

# #1_6.[ x데이터 컬럼 정리 및 shape 확인 ]###################################################################

탑건 = 탑건.drop(['일자', '매출액', '날짜'], axis=1)
아이언맨 = 아이언맨.drop(['일자', '매출액', '날짜'], axis=1)
어벤져스 = 어벤져스.drop(['일자', '매출액', '날짜'], axis=1)
엔드게임 = 엔드게임.drop(['일자', '매출액', '날짜'], axis=1)
닥터스트레인지 = 닥터스트레인지.drop(['일자', '매출액', '날짜'], axis=1)
토르 = 토르.drop(['일자', '매출액', '날짜'], axis=1)

# print(탑건.shape)             # (31, 8)
# print(아이언맨.shape)         # (118, 8)
# print(어벤져스.shape)         # (118, 8)
# print(엔드게임.shape)         # (118, 8)
# print(닥터스트레인지.shape)   # (31, 8)
# print(토르.shape)             # (31, 8)


#1_7.[ 데이터 정규화 ]###################################################################

from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

scaler = MaxAbsScaler()

# 정규화 대상 column 정의
scale_cols = ['상영횟수', '좌석수', '관객수', '누적매출액', '누적관객수', '연도', '월', '일']

탑건_scaler = scaler.fit_transform(탑건[scale_cols])
아이언맨_scaler = scaler.fit_transform(아이언맨[scale_cols])
어벤져스_scaler = scaler.fit_transform(어벤져스[scale_cols])
엔드게임_scaler = scaler.fit_transform(엔드게임[scale_cols])
닥터스트레인지_scaler = scaler.fit_transform(닥터스트레인지[scale_cols])
토르_scaler = scaler.fit_transform(토르[scale_cols])

# print(탑건_scaler)
# print(아이언맨_scaler)
# print(어벤져스_scaler)
# print(엔드게임_scaler)
# print(닥터스트레인지_scaler)
# print(토르_scaler)


#1_8[DataFrame을 numpy로 변환 작업 ]#############################################################

탑건_scaler = pd.DataFrame(탑건_scaler, columns=scale_cols)
탑건_scaler = 탑건_scaler.to_numpy()
탑건_매출액 = 탑건_매출액.to_numpy()

아이언맨_scaler = pd.DataFrame(아이언맨_scaler, columns=scale_cols)
아이언맨_scaler = 아이언맨_scaler.to_numpy()
아이언맨_매출액 = 아이언맨_매출액.to_numpy()

어벤져스_scaler = pd.DataFrame(어벤져스_scaler, columns=scale_cols)
어벤져스_scaler = 어벤져스_scaler.to_numpy()
어벤져스_매출액 = 어벤져스_매출액.to_numpy()

엔드게임_scaler = pd.DataFrame(엔드게임_scaler, columns=scale_cols)
엔드게임_scaler = 엔드게임_scaler.to_numpy()
엔드게임_매출액 = 엔드게임_매출액.to_numpy()

닥터스트레인지_scaler = pd.DataFrame(닥터스트레인지_scaler, columns=scale_cols)
닥터스트레인지_scaler = 닥터스트레인지_scaler.to_numpy()
닥터스트레인지_매출액 =닥터스트레인지_매출액.to_numpy()

토르_scaler = pd.DataFrame(토르_scaler, columns=scale_cols)
토르_scaler = 토르_scaler.to_numpy()
토르_매출액 = 토르_매출액.to_numpy()

# #1_9[ 훈련 데이터 train, test 분리 ]#############################################################################
from sklearn.model_selection import train_test_split

아이언맨_x_train, 아이언맨_x_test, 아이언맨_y_train, 아이언맨_y_test = train_test_split(아이언맨_scaler, 아이언맨_매출액,
                                                                            train_size=0.8,
                                                                            shuffle=False
                                                                            )
어벤져스_x_train, 어벤져스_x_test, 어벤져스_y_train, 어벤져스_y_test = train_test_split(어벤져스_scaler, 어벤져스_매출액,
                                                                            train_size=0.8,
                                                                            shuffle=False
                                                                            )
엔드게임_x_train, 엔드게임_x_test, 엔드게임_y_train, 엔드게임_y_test = train_test_split(엔드게임_scaler, 엔드게임_매출액,
                                                                            train_size=0.8,
                                                                            shuffle=False
                                                                            )



#1_10[ 3차원으로 변환 ]####################################################################################################################3


아이언맨_x_train = 아이언맨_x_train.reshape(94, 8, 1)  
아이언맨_x_test = 아이언맨_x_test.reshape(24, 8, 1)
아이언맨_y_train = 아이언맨_y_train.reshape(94, 1)
아이언맨_y_test = 아이언맨_y_test.reshape(24, 1)
# print(아이언맨_x_train.shape)
# print(아이언맨_x_test.shape)
# print(아이언맨_y_train.shape)
# print(아이언맨_y_test.shape)


어벤져스_x_train = 어벤져스_x_train.reshape(94, 8, 1)  
어벤져스_x_test = 어벤져스_x_test.reshape(24, 8, 1)
어벤져스_y_train = 어벤져스_y_train.reshape(94, 1)
어벤져스_y_test = 어벤져스_y_test.reshape(24, 1)
# print(어벤져스_x_train.shape)
# print(어벤져스_x_test.shape)
# print(어벤져스_y_train.shape)
# print(어벤져스_y_test.shape)


엔드게임_x_train = 엔드게임_x_train.reshape(94, 8, 1)  
엔드게임_x_test = 엔드게임_x_test.reshape(24, 8, 1)
엔드게임_y_train = 엔드게임_y_train.reshape(94, 1)
엔드게임_y_test = 엔드게임_y_test.reshape(24, 1)
# print(엔드게임_x_train.shape)
# print(엔드게임_x_test.shape)
# print(엔드게임_y_train.shape)
# print(엔드게임_y_test.shape)


탑건_x_test = 탑건_scaler.reshape(31, 8, 1)  
탑건_y_test = 탑건_매출액.reshape(31, 1)
# print(토르_scaler.shape)
# print(토르_매출액.shape)


닥터스트레인지_x_test = 닥터스트레인지_scaler.reshape(31, 8, 1)  
닥터스트레인지_y_test = 닥터스트레인지_매출액.reshape(31, 1)
# print(닥터스트레인지_scaler.shape)
# print(닥터스트레인지_매출액.shape)


토르_x_test = 토르_scaler.reshape(31, 8, 1)  
토르_y_test = 토르_매출액.reshape(31, 1)
# print(토르_scaler.shape)
# print(토르_매출액.shape)


#2.[ 모델구성 ]###########################################################################################
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, GRU, RNN
from tensorflow.python.keras.models import Input, Model

#2-1 모델 아이언맨
아이언맨_input = Input(shape=(8, 1))     
dense12 = GRU(128, activation='relu')(아이언맨_input)
dense22 = Dense(128, activation='relu')(dense12)
dense22 = Dropout(0.2)(dense22)
dense32= Dense(64, activation='relu')(dense22)
dense22 = Dropout(0.2)(dense32)
dense42= Dense(128, activation='relu')(dense22)
아이언맨_mid = Dense(64, activation='relu')(dense42)

#2-2 모델 어벤져스
어벤져스_input = Input(shape=(8, 1))    
dense13 = GRU(128, activation='relu')(어벤져스_input)
dense23 = Dense(128, activation='relu')(dense13)
dense23 = Dropout(0.2)(dense23)
dense33= Dense(64, activation='relu')(dense23)
dense33 = Dropout(0.2)(dense33)
dense43= Dense(128, activation='relu')(dense33)
어벤져스_mid = Dense(64, activation='relu')(dense43)

#2-3 모델 엔드게임
엔드게임_input = Input(shape=(8, 1))    
dense14 = GRU(128, activation='relu')(엔드게임_input)
dense24 = Dense(128, activation='relu')(dense14)
dense24 = Dropout(0.2)(dense24)
dense34= Dense(64, activation='relu')(dense24)
dense24 = Dropout(0.2)(dense24)
dense44= Dense(128, activation='relu')(dense34)
엔드게임_mid = Dense(64, activation='relu')(dense44)


#2_4 모델 엮기
from tensorflow.python.keras.layers import concatenate
merge1 = concatenate([아이언맨_mid, 어벤져스_mid, 엔드게임_mid], name='mg1')
merge2 = Dense(128, activation='relu', name='mg2_15')(merge1)
merge2 = Dense(128, activation='relu', name='mg2_16')(merge1)
merge2 = Dense(356, activation='relu', name='mg2_17')(merge1)
merge2 = Dense(128, activation='relu', name='mg2_18')(merge1)
merge3 = Dense(64, name='mg3_12')(merge2)
concatenate_output = Dense(32, name='last')(merge3)

#2_5 모델 분리
output12 =  Dense(100)(concatenate_output)
output22 = Dense(100)(output12)
아이언맨_output = Dense(1, name='last2')(output22)

output13 =  Dense(100)(concatenate_output)
output23 = Dense(100)(output13)
어벤져스_output = Dense(1, name='last3')(output23)

output14 =  Dense(100)(concatenate_output)
output24 = Dense(100)(output14)
엔드게임_output = Dense(1, name='last4')(output24)

#2_6 모델 아웃풋 빼기
from tensorflow.python.keras.models import Model
model = Model(inputs=[아이언맨_input, 어벤져스_input, 엔드게임_input],
              outputs=[아이언맨_output, 어벤져스_output, 엔드게임_output])
# model.summary()



#3.[ 컴파일, 훈련 ]###################################################################################################
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([filepath, 'k24_', date, '_', filename])
                      )

hist = model.fit([아이언맨_x_train, 어벤져스_x_train, 엔드게임_x_train],
                 [아이언맨_y_train, 어벤져스_y_train, 엔드게임_y_train],
                 epochs=10000,
                 batch_size=5,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)
model.save('./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')

from tensorflow.python.keras.models import load_model
model = load_model("./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5")


# 4.[ 평가 및 예측 ]##################################################################################################################
loss = model.evaluate([아이언맨_x_test , 어벤져스_x_test, 엔드게임_x_test, ], [아이언맨_y_test , 어벤져스_y_test, 엔드게임_y_test ])  
print("loss :  ",loss)

y_predict1, y_predict2, y_predict3 = model.predict([토르_x_test , 탑건_x_test , 닥터스트레인지_x_test])
# y_predict1, y_predict2, y_predict3 = model.predict([아이언맨_x_test , 어벤져스_x_test , 엔드게임_x_test])
from sklearn.metrics import r2_score
y1_r2 = r2_score(토르_y_test , y_predict1)   
y2_r2 = r2_score(탑건_y_test , y_predict2)   
y3_r2 = r2_score(닥터스트레인지_y_test, y_predict3)   
print("r2_score",y1_r2)
print("r2_score",y2_r2)
print("r2_score",y3_r2)
print("토르 08.05 매출가 : ",y_predict1[-1])



