"""=[ .save 설명 ]==============================================================================================

model.save("./_save/keras23_1_save_model.h5")  <-- 사용자가 원하는 경로로 넣어서 사용하면 된다.
# 모델에 관련 정보를 해당 경로에 세이브 해둠















===[ 모델구성만 세이브 ]==============================================================================================


#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation ='relu'))
model.add(Dense(16, activation ='relu'))
model.add(Dense(8, activation ='relu'))
model.add(Dense(1))

model.save("./_save/keras23_1_save_model.h5")
# 아래에서 사용할 경우 모델구성만 세이브

















===[ 모델과 weight까지 세이브 ]==============================================================================================

#3. 컴파일. 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True) 

start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=50, verbose=1, validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time() -start_time

model.save("./_save/keras23_1_save_model.h5")
# fit 단계 다음에 해주면 모델과 weight 까지 저장됨

========================================================================================================================
"""   

























































































































"""=[ load_model 설명 ]==============================================================================================

# 해당 경로에 있는 모델을 불러옴
# 해당 경로에 있는 저장된 모델과 가중치 값을 가져온다
# 해당 모델을 만들 때 나왔던 가중치 값이 그대로 저장되고 그 값을 가져옴

model = load_model('저장경로') 
fit 다음에 save 해준 모델을 컴파일, 훈련단계 위에서 다시 불러올 경우 해당 모델에서 weight값이 새로 구해지지만
불러온 weight에 덮어 씌워짐

그래서 제일 좋게 나온 weight 값을 그것만 따로 불러오려면 fit 다음에 불러와야 함 


















===[ load_model 사용 ]==============================================================================================

#3. 컴파일, 훈련
)        
# model.fit(x_train, y_train, epochs=3000, batch_size=100,
#                  validation_split=0.2,
#                  callbacks=[earlyStopping],
#                  verbose=1)


start_time = time.time()

model = load_model("./_save/keras23_3_save_model.h5")

end_time = time.time() - start_time   # 해당 모델의 시간을 불러와줌
























--[ load_model의 weight값 갱신하는 방법 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

이미 저장된 상태에서 모델 아래쪽에 위치시키고 컴파일, 훈련을 적용시켜주면
저장된 파일에서 새로운 가중치(랜덤)로 값 갱신 

model = load_model("./_save/keras23_3_save_model.h5")  # 가중치와 모델구성을 세이브 해놨던 모델을 불러옴

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping =EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                             restore_best_weights=True) 

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=20, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음

end_time = time.time() - start_time

model.save("./_save/keras23_3_save_model.h5")            # 기존 모델에서 새로운 가중치 값이 들어감
                                                          [중요]기존의 가중치 값이 더 좋으면 가중치 변경 안함

model = load_model("./_save/keras23_3_save_model.h5")    # ex)로 위에서 세이브한 모델을 바로 가져와서 바로 평가에 사용

========================================================================================================================
"""   