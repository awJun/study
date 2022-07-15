#1. 데이터
import numpy as np       # v-- 삼성        # v-- 하이닉스
x1_datasets = np.array([range(100), range(301, 401)])   # 삼성전자, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]) # 원유, 돈육, 밀
x3_datasets = np.array([range(100, 200), range(1301, 1401)])   # 우리반 아이큐, 우리반 키
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

# print(x1.shape, x2.shape, x3.shape) # (100, 2) (100, 3) (100, 2)

y = np.array(range(2001, 2101))  #(100,)    # 금리


# x1와 x2의 관한 y의 모델


from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(x1, x2, x3, y,
                                                                        train_size=0.7,
                                                                        shuffle=True,
                                                                        random_state=100
                                                                        )
# print(x1_train.shape, x1_test.shape)   # (70, 2) (30, 2)
# print(x2_train.shape, x2_test.shape)   # (70, 3) (30, 3)
# print(x3_train.shape, x3_test.shape)   # (70, 2) (30, 2)
# print(y_train.shape, y_test.shape)     # (70,) (30,)

#2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input

#2-1. 모델 x1
input1 = Input(shape=(2,))    # print(x1_train.shape, x1_test.shape)   # (70, 2) (30, 2)
dense1 = Dense(100, activation='relu', name='jun1')(input1)
dense2 = Dense(200, activation='relu', name='jun2')(dense1)
dense3= Dense(300, activation='relu', name='jun3')(dense2)
output1 = Dense(100, activation='relu', name='out_jun1')(dense3)


#2-2 모델 x2
input2 = Input(shape=(3,))     # print(x2_train.shape, x2_test.shape)   # (70, 3) (30, 3)
dense11 = Dense(110, activation='relu', name='jun11')(input2)
dense12 = Dense(120, activation='relu', name='jun12')(dense11)
dense13= Dense(130, activation='relu', name='jun13')(dense12)
dense14= Dense(140, activation='relu', name='jun14')(dense13)
output2 = Dense(100, activation='relu', name='out_jun2')(dense14)

#2-2 모델 x3
input3 = Input(shape=(2,))     # print(x2_train.shape, x2_test.shape)   # (70, 2) (30, 2)
dense111 = Dense(100, activation='relu', name='jun111')(input3)
dense112 = Dense(100, activation='relu', name='jun112')(dense111)
dense113= Dense(100, activation='relu', name='jun113')(dense112)
dense114= Dense(100, activation='relu', name='jun114')(dense113)
output3 = Dense(100, activation='relu', name='out_jun3')(dense114)

#2-3 모델 concate(엮기)
from tensorflow.python.keras.layers import concatenate, Concatenate  # <- 그냥 같이 붙여버리는 개념
# concate : 엮다         <--리스트에서 .append 개념이라 생각하면 된다.
merge1 = concatenate([output1, output2, output3], name='mg1')
merge2 = Dense(200, activation='relu', name='mg2')(merge1)
merge3 = Dense(30, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)   # <-- 레이어이므로 마지막은 1

model = Model(inputs=[input1, input2, input3], outputs=last_output)

    
 
#3. 컴파일. 훈련
model.compile(loss='mae', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)        

start_time = time.time()
model.fit([x1_train, x2_train, x3_train], y_train, epochs=10000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() -start_time



#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)

y_predict = model.predict([x1_test, x2_test, x3_test])

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   

print('loss : ', loss)
print('r2스코어 : ', r2)
print('걸린시간 : ', end_time) 
 

# loss :  7.333516597747803
# r2스코어 :  0.9163418016368319
# 걸린시간 :  4.783814191818237

























