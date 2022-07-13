"""=[ .save_weights 설명 ]==============================================================================================

#### model.save_weights('저장경로')
# fit 단계 다음에 해줘야 함

#### model.load_weights('저장경로')
# fit 다음에 save한 파일을 모델 밑에다 불러와주면 됨
# 얘는 훈련한 다음의 가중치가 저장 돼 있어서 loss와 r2가 동일하게 나옴 (3단계에서 컴파일만 해주면 됨)

# save_weights, load_weights는 일반 save와 다르게 model = Sequential()과 model.compile()해줘야 사용이 가능함 
# 저장된 weights를 불러올 때는 모델구성, compile을 해주면 됨 (fit 생략)
# fit단계 전에 하냐 후에 하냐에 따라 차이가 있지만 후에 쓰는게 바른 방법이고 그래야 값이 저장됨












===[ .save_weights 사용법 ]==============================================================================================

model = Sequential()                             # _weights에서는 없으면 사용불가
model.compile(loss='mse', optimizer='adam')      # _weights에서는 없으면 사용불가

hist = model.fit(x_train, y_train, epochs=100, batch_size=20, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1) #verbose=0 일때는 훈련과정을 보여주지 않음

model.save_weights("./_save/keras23_5_save_weights2.h5")


















===[ .load_weights 사용법 ]==============================================================================================

model = Sequential()                             # _weights에서는 없으면 사용불가
model.compile(loss='mse', optimizer='adam')      # _weights에서는 없으면 사용불가

model.load_weights("./_save/keras23_5_save_weights2.h5")  

========================================================================================================================
"""   