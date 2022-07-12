import numpy as np
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_zzz, y_train, y_zzz = train_test_split(x, y,
                                                    train_size=0.625,
                                                    shuffle=True, 
                                                    random_state=100                
                                                                     )
x_test, y_test, x_val, y_val = train_test_split(x_zzz, y_zzz,
                                                    train_size=0.5,
                                                    shuffle=True, 
                                                    random_state=100                
                                                                     )

print(x_train)  # x_train
print(y_train)  # y_train
print(x_test) # x_test
print(y_test)   # y_test
print(x_val)    # x_val
print(y_val)    # x_val






#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.25)
# validation_split=0.25 데이터의 0.25를 validation로 빼서 사용하겠다. 라는 뜻

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

result = model.predict([17])
print("17의 예측값 : ", result)















