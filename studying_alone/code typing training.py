import numpy as np
import time

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = accuracy_score()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation = 'relu', input_dim=30))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor='val_loss', mode = 'auto', patience=100,
                              verbose=1,
                              restore_best_weights=True)

start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=100, validation_split=0.2,
          callback=[earlystopping])
end_time = time.time()

#4. 평가, 예측
y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)

acc = accuracy_score(y_test, y_predict)

print("loss : ", loss)
print("accuracy : ", acc)
print("걸린시간 : ", end_time)

















































































