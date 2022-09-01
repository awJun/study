# 훈련시간 닶없다..... 쓰지말자 ㅠㅠ


import pandas as pd
import numpy as np
import glob

path = './_data/dacon_Bok/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8

def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)
    print('끗.')

x_train, y_train = aaa(train_input_list, train_target_list) #, False)
x_test, y_test = aaa(val_input_list, val_target_list) #, False)

# print(x_train.shape)  # (1607, 1440, 37)
# print(y_train.shape)  # (1607,)
# print(x_test.shape)   # (2019, 1440, 37)
# print(y_test.shape)   # (2019,)

#--[ 스케일러 작업]- - - - - - - - - - - - - - - - - -(데이터 안에 값들의 차이을 줄여줌(평균으로 만들어주는 작업))
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

x_train = x_train.reshape(1607, 1440 * 37)              
x_test = x_test.reshape(2019, 1440 * 37)

scaler =  MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
                                
scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) 
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[차원 변형 작업]- - - - - - - - - - - - - - - - - - - - - - - 
x_train = x_train.reshape(1607, 1440, 37)              
x_test = x_test.reshape(2019, 1440, 37)

print(x_train.shape)  # (10, 3, 1)  <-- "2, 2 ,1"는 input_shape값
print(x_test.shape)   # (3, 3, 1)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM 
model = Sequential()                            # input_shape=(3, 1) == input_length=3, input_dim=1)
# model.add(SimpleRNN(units=100,activation='relu' ,input_shape=(3, 1)))   # [batch, timesteps, feature]

model.add(LSTM(units=128 ,input_length=1440, input_dim=37))   #SimpleRNN를 거치면 3차원이 2차원으로 간다.
# model.add(SimpleRNN(10))       # <-- 3차원이 아니라 2차원을 넣어서 에러가 발생함.[참고]
model.add(Dense(128, activation='relu'))
model.add(Dense(356, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1,
                              restore_best_weights=True) 

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   
print("r2 스코어 : ", r2)

# y_pred = np.array([8, 9, 10]).reshape(1, 3, 1)      #8, 9, 10을 넣어서 11일을 예측       # [중요]rnn 모델에서 사용할 것이므로 3차원으로 변환작업
#                                                     # .reshape 앞에 array([8, 9, 10])를 (1, 3, 1)로 바꾸겟다. [[[8], [9], [10]]]

# y_pred안에 np.array([8, 9, 10]) 배열이 3개의 값이 들어 있으므로 

# .reshape(1, 3, 1) 안에 1, 3, 1인 이유는 x.reshape(7, 3, 1)에서 3, 1 부분을  input_shape=(3, 1)에 넣어서 사용해서 3, 1 부분을
  # 넣고 뒤에 1을 곱하는 형식으로 3차춴으로 만들어 줬다
# result = model.predict(y_pred) 
# print("loss : ", loss)
# print("[8,9,10의 결과", result)


# import os
# import zipfile
# filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
# os.chdir("D:/study_data/_data\dacon_Bok/sample_submission")
# with zipfile.ZipFile("submission.zip", 'w') as my_zip:
#     for i in filelist:
#         my_zip.write(i)
#     my_zip.close()



###[ r2_score ]###########################
# 결과:  0.8070573257161597











