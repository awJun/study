# [SimpleRNN] units : 10 -> 10* (1 + 1 +10) = 120

# [LSTM] units : 10 -> 4 * 10 * (1 + 1 + 10) = 480
                    #    4 * 20 * (1 + 1 + 20) = 1760
# [결론] LSTM = simpleRNN * 4
# 숫자4의 의미는 cell state, input gate, output gate, forget gate
# 한마디로 LSTM이 SimpleRNN보다  연산량이 많다 성능도 좋다 하지만 안좋은 경우도 있다 둘 다 사용해보고 알아서 튜닝할 것 ㅋ

# [GPU] units : 10 -> 3  10 * (1 + 1 + 10) = 360

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# model.add(SimpleRNN(units=100 ,input_length=3, input_dim=1))
#                     SimpleRNN -->  LSTM로 바꾸는 방법 그냥 앞에꺼만 바꾸면 됨 ㅋ
# model.add(LSTM(units=100 ,input_length=3, input_dim=1))




# LSTM가 SimpleRNN보다 성능이 좋다 (그만큼 연산량이 많다.)
# LSTM 설명 링크  https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=magnking&logNo=221311273459

#=======================================================================================================================

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM,  GRU
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split # 함수 가능성이 높음

#1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
             [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
             [9, 10, 11], [10, 11, 12],
             [20, 30, 40], [30, 40, 50], [40, 50, 60]]
             )
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])        # 80이라는 숫자가 나오도록 예측하자 ~      50, 60, 70을 예측에 넣으면 될거같음

# print(x.shape)  (13, 3)
# print(y.shape)  (13,)
#                                                #input_dim = feature
 # 2차원 데이터를 3차원으로 만들 기 위해 뒤에 1을 곱하는 형태로 만들어줌 (1을 곱하면 데이터값 변동이 없기 때문)
 
                                                # SimpleRNN를 사용하려면 input이 무조건 3차원이어야 한다.
     
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )


# print(x_train.shape)   # (10, 3)
# print(y_train.shape)   # (10, )
# print(x_test.shape)    # (3, 3)
# print(y_test.shape)    # (3,)

#--[ 스케일러 작업]- - - - - - - - - - - - - - - - - -(데이터 안에 값들의 차이을 줄여줌(평균으로 만들어주는 작업))
# scaler =  MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
                                
scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[차원 변형 작업]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
x_train = x_train.reshape(10, 3, 1)              
x_test = x_test.reshape(3, 3, 1)

# print(x_train.shape)  # (10, 3, 1)  <-- "2, 2 ,1"는 input_shape값
# print(x_test.shape)   # (3, 3, 1)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

                                          
                                                
#2. 모델구성 
model = Sequential()                            # input_shape=(3, 1) == input_length=3, input_dim=1)
# model.add(SimpleRNN(units=100,activation='relu' ,input_shape=(3, 1)))   # [batch, timesteps, feature]

model.add(GRU(units=100 ,input_length=3, input_dim=1))   #SimpleRNN  또는 LSTM를 거치면 3차원이 2차원으로 변형되어서 다음 레어어에 간다.
# model.add(SimpleRNN(10))       # <-- 3차원이 아니라 2차원을 넣어서 에러가 발생함.[참고]
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.summary()  # https://velog.io/@yelim421/%EC%88%9C%ED%99%98-%EC%8B%A0%EA%B2%BD%EB%A7%9D-Recurrent-Neural-NetworkRNN / Param가 120인 이유 연산과정



# #3. 컴파일, 훈련
# from tensorflow.python.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1,
#                               restore_best_weights=True) 

# model.compile(loss='mse', optimizer='adam')

# from tensorflow.python.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1,
#                               restore_best_weights=True) 

# model.fit(x_train, y_train, epochs=10000, batch_size=32, verbose=1, 
#                  callbacks=[earlyStopping])  

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# y_pred = np.array([50, 60, 70]).reshape(1, 3, 1)      #8, 9, 10을 넣어서 11일을 예측       # [중요]rnn 모델에서 사용할 것이므로 3차원으로 변환작업
#                                                     # .reshape 앞에 array([8, 9, 10])를 (1, 3, 1)로 바꾸겟다. [[[8], [9], [10]]]

# # y_pred안에 np.array([8, 9, 10]) 배열이 3개의 값이 들어 있으므로 

# # .reshape(1, 3, 1) 안에 1, 3, 1인 이유는 x.reshape(13, 3, 1)에서 3, 1 부분을  input_shape=(3, 1)에 넣어서 사용해서 3, 1 부분을
#   # 넣고 앞에 1을 곱하는 형식으로 3차춴으로 만들어 줬다
# result = model.predict(y_pred) 
# print("loss : ", loss)
# print("[50, 60, 70의 결과 : ", result)


# # = = = = = = = = = = = = = = = = = = = = = =
# # scaler = RobustScaler

# # loss :  2.8614425659179688
# # [50, 60, 70의 결과 :  [[78.02161]]
# # = = = = = = = = = = = = = = = = = = = = = =
# # scaler = StandardScaler

# # loss :  1.8919610977172852
# # [50, 60, 70의 결과 :  [[86.888596]]
# # = = = = = = = = = = = = = = = = = = = = = =



# # [GPU] units : 10 -> 3  10 * (1 + 1 + 10) = 360


# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# gru (GRU)                    (None, 100)               30600     
# _________________________________________________________________
# dense (Dense)                (None, 100)               10100     
# _________________________________________________________________
# dense_1 (Dense)              (None, 300)               30300     
# _________________________________________________________________
# dense_2 (Dense)              (None, 300)               90300     
# _________________________________________________________________
# dense_3 (Dense)              (None, 300)               90300     
# _________________________________________________________________
# dense_4 (Dense)              (None, 300)               90300     
# _________________________________________________________________
# dense_5 (Dense)              (None, 300)               90300     
# _________________________________________________________________
# dense_6 (Dense)              (None, 100)               30100     
# _________________________________________________________________
# dense_7 (Dense)              (None, 1)                 101       
# =================================================================
# Total params: 462,401
# Trainable params: 462,401
# Non-trainable params: 0
# _________________________________________________________________
