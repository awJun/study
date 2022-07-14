import numpy as np
from sklearn import datasets

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split # 함수 가능성이 높음


trainset = np.array(range(1,101))  # range(1, 11-1)           # print(dataset)  # [ 1  ~ 100 ]
testset = np.array(range(96,106))  # (96 ~ 105)               # 96, 97, 98, 99에대한 예측 이런씩으로 자르기

# print(len(trainset))

def split_x(a, b):
    임의의_리스트선언 = []
    for i in range(len(a) - b + 1):   # 10 - 5 + 1  =6 /  i가 반복하는 건 6번  /  0 1 2 3 4 5 <--  i가 들어가는 값의 순서
        subset = a[i : (i + b)]    #  subset = dataset[0 : 5]   1, 2, 3, 4, 5 
        임의의_리스트선언.append(subset)   
    
       
    return np.array(임의의_리스트선언)

#--[trainset 만드는 과정]-----------
size = 5   #  x = 4개  y는 1개
train = split_x(trainset, size)

# print(train)         
# print(train.shape)    # (96, 5)
#-----------------------------------


#---[trainset에서 x와 y데이터를 추출하는 과정]---------------------------------------------------------------------------------------------------------
# 1, 2, 3, 4를 x데이터로 만드는 과정 5번째 열은 뺌 / 왜냐하면 "1, 2, 3, 4에 대한 예측은 5"라는 형태의 데이터로 만들기 위해 y에서 사용할 것이기 때문이다.
x = train[:, :-1]     
# x에서 빼버린 5번째 열을 y데이터로 사용하겟다는 뜻 ~ 
y = train[:, -1]       
#-----------------------------------------------------------------------------------------------------------------------------------------------------


#-[예측 단계중 predict에서 사용 할 데이터 만드는 과정]- - - - - - - - - - -
size = 4
test = split_x(testset, size)

# print(test)
# print(x.shape, y.shape, test.shape)

#- - - - - - - - - - - - - - - - -

# 모델 구성 및 평가 예측할 것.

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )


# print(x_train.shape)   # (76, 4)
# print(y_train.shape)   # (76,)
# print(x_test.shape)    # (20, 4)
# print(y_test.shape)    # (20,)

#--[ 스케일러 작업]- - - - - - - - - - - - - - - - - -(데이터 안에 값들의 차이을 줄여줌(평균으로 만들어주는 작업))
# scaler =  MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
                                
scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[차원 변형 작업]- - - - - - - - - - - - - - - - - - - - - - - - - - 
x_train = x_train.reshape(76, 4, 1)              
x_test = x_test.reshape(20, 4, 1)

print(x_train.shape)  # (76, 4, 1)  <-- "4, 1"는 input_shape값
print(x_test.shape)   # (20, 4, 1)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

                                          
                                                
#2. 모델구성 
model = Sequential()                            # input_shape=(3, 1) == input_length=3, input_dim=1)
# model.add(SimpleRNN(units=100,activation='relu' ,input_shape=(3, 1)))   # [batch, timesteps, feature]

model.add(LSTM(units=100 ,input_length=4, input_dim=1))   #SimpleRNN  또는 LSTM를 거치면 3차원이 2차원으로 변형되어서 다음 레어어에 간다.
# model.add(SimpleRNN(10))       # <-- 3차원이 아니라 2차원을 넣어서 에러가 발생함.[참고]
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
# model.summary()  # https://velog.io/@yelim421/%EC%88%9C%ED%99%98-%EC%8B%A0%EA%B2%BD%EB%A7%9D-Recurrent-Neural-NetworkRNN / Param가 120인 이유 연산과정



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1,
                              restore_best_weights=True) 

model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1,   # batch_size의 디폴트값 = 32 즉 32이는 넣으나 마나임 ~
                 callbacks=[earlyStopping])  

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_pred = test.reshape(-1, 4, 1)      #8, 9, 10을 넣어서 11일을 예측       # [중요]rnn 모델에서 사용할 것이므로 3차원으로 변환작업
                                                    # .reshape 앞에 array([8, 9, 10])를 (1, 3, 1)로 바꾸겟다. [[[8], [9], [10]]]

        # (-1, 4, 1)  == (데이터양, 행, 열)  -1를 넣으면 자동으로 해준다. 즉 aoto기능 ~

# y_pred안에 np.array([8, 9, 10]) 배열이 3개의 값이 들어 있으므로 

# .reshape(1, 3, 1) 안에 1, 3, 1인 이유는 x.reshape(13, 3, 1)에서 3, 1 부분을  input_shape=(3, 1)에 넣어서 사용해서 3, 1 부분을
  # 넣고 앞에 1을 곱하는 형식으로 3차춴으로 만들어 줬다
result = model.predict(y_pred) 
print("loss : ", loss)
print("test의 결과 : ", result)



















































