import numpy as np
import time

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import fetch_covtype

#.1 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape)       # (581012, 54)
print(y.shape)       # (581012,)
print(np.unique(y, return_counts=True))   # (1, 2, 3, 4, 5, 6, 7)


from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
print(y)
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )


#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=54))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(7, activation='softmax')) 

#3. 컴파일. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy']) 

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1, batch_size=100,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time()



#4. 평가, 예측
import tensorflow as tf
################################################################################
loss, acc = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력
                                            # loss = loss / acc = metrics에서 나온 accuracy 값
print('loss : ', loss)
print('acc : ', acc)


result = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

y_predict = tf.argmax(y_predict, axis=1)
print(y_predict)

y_test = tf.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)



















