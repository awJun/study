import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))   # 1 * 3
model.add(Dense(2))         
model.add(Dense(1))
model.add(Dense(1))

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))
print(model.weights)
print("==========================================")

print(model.trainable_weights)
print("==========================================")

model.trainable=False  # Trainable params: 23 훈련된 데이터

print("===========================================")
print(len(model.weights))
print(len(model.trainable_weights))

print("===========================================")
print(model.trainable_weights)

model.summary()

model.compile(loss='mse', optimizer='adam')
# model.fit(x, y, batch_size=1, epochs=100)

# y_predict = model.predict(x)
# print(y_predict[:3])
# [[0.59276265]
#  [1.1855253 ]
#  [1.778288  ]]








