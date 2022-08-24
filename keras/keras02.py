import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np.array([1, 2, 3, 5, 4])
y = np.array([1, 2, 3, 4, 5])

# 2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(23)) 
model.add(Dense(30))  
model.add(Dense(30))  
model.add(Dense(40))
model.add(Dense(50)) 
model.add(Dense(80)) 
model.add(Dense(80))  
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. 평가, 예측
 #평가
loss = model.evaluate(x, y) 
print('loss : ', loss)

 #예측
result = model.predict([6])
print('6의 예측값 : ', result) 
  
            
