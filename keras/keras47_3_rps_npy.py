# 1에서 넘파이를 불러와서 모델링
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

x = np.load('D:/study_data/_save/_npy/rps/keras47_3_x_data.npy')
y = np.load('D:/study_data/_save/_npy/rps/keras47_3_y_data.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

print(x_train.shape)   # (2016, 150, 150, 3)                                           
print(x_test.shape)    # (504, 150, 150, 3)                                              
print(y_train.shape)   # (504, 150, 150, 3)                                                
print(y_test.shape)    # (504,)                                          
    
#2.모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape = (150, 150, 3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='softmax'))
# model.summary()


#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam')
# batch를 최대로 잡으면 이렇게도 가능.
from tensorflow.python.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor= 'val_loss', patience=30, mode='auto', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, 
                 validation_split=0.2, 
                 verbose=1, 
                 batch_size=32,
                 callbacks=es
                 )


#4.평가,예측 
loss = model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)   
y_predict = np.argmax(y_predict, axis= 1)
# print(y_predict)

from sklearn.metrics import accuracy_score

y_test = np.argmax(y_test, axis= 1)
# print(y_test)
acc = accuracy_score(y_test,y_predict)

print('loss : ', loss)
print('acc : ', acc)


















