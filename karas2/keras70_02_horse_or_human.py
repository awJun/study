# 1에서 넘파이를 불러와서 모델링

import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

x = np.load('D:/study_data/_save/_npy/horse_or_human/keras47_2_x_data.npy')
y = np.load('D:/study_data/_save/_npy/horse_or_human/keras47_2_y_data.npy')

# print(x.shape)    # (1027, 150, 150, 3)
# print(y.shape)    #  (1027,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )


#2. 모델구성
from keras.applications import VGG19
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten
from sklearn.metrics import r2_score,accuracy_score
keras_model = VGG19(weights="imagenet", include_top=False,  
                input_shape=(150, 150, 3))

model = Sequential()
model.add(keras_model) 
model.add(Flatten())
model.add(Dense(128, activation='relu'))  
model.add(Dense(64, activation='relu'))  
model.add(Dense(1, activation='sigmoid'))
# model.summary()


#3.컴파일,훈련
model.compile(loss= 'binary_crossentropy',optimizer='adam')
# batch를 최대로 잡으면 이렇게도 가능.
from tensorflow.python.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor= 'val_loss', patience=30, mode='auto', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, 
                 validation_split=0.2, 
                 verbose=1, 
                 batch_size=32,
                 callbacks=es)


#4.평가,예측 
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)   

from sklearn.metrics import accuracy_score
y_predict = y_predict.round()
acc = accuracy_score(y_test,y_predict)

print('loss : ', loss)
print('acc : ', acc)



# loss :  0.0026676952838897705
# acc :  1.0











