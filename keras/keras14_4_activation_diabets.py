import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import load_diabetes

datasets = load_diabetes()

x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=72
                                                    )



#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=10))
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




#4. 평가, 예측
y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print('r2스코어 : ', r2)
print("걸린시간 : ", end_time)


# loss :  2209.875244140625
# r2스코어 :  0.6311260734412032
# 걸린시간 :  1656748896.5937438






























