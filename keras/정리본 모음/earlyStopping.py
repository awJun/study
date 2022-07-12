#3. 컴파일. 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1,
                              restore_best_weights=True) 

# 모니터를 하다가 var_loss를 중심으로 중지를 시키겠다.
# val_loss가 10번까지  최소값이 올라가면 중지를 하겠다.  (최소값 이후로 최소값보다 넘어간 횟수가 10번 넘어가면 정지)
# restore_best_weights=True 를 True로하면 최소값이 10번 올라가기 전의 값을 사용한다는 뜻이다.

# mode='min'를 오토로 두면 상황에 맞춰서 최대값, 최소값으로 상황에 맞게 사용해준다.
# r2 같은 경우는 값이 올라가기 때문에 최대값으로 사용해야한다.
# mode을 설정하기 귀찮으면 그냥 auto로 마추면 알아서 상황에 맞게 설정해줌


model.fit(x_train, y_train, epochs=10000, batch_size=100, verbose=1, validation_split=0.2,
                 callbacks=[earlyStopping])  
"""
Epoch 00033: early stopping 이라는 문구가 뜨면서 정지함.
"""  