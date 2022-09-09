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
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse','accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor= 'val_loss', patience=30, mode='auto', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, 
                 validation_split=0.2, 
                 verbose=1, 
                 batch_size=32,
                 callbacks=es)

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss :', loss[-1])       
print('val_loss :', val_loss[-1])
print('accuracy :', accuracy[-1])
print('val_accuracy :', val_accuracy[-1])   # 위에서 history를 사용해서 값이 여러개가 출력된다 그중 제일 최근의 로스인 마지막 부분을 가져와서 출력했다.


# loss : 1.0986337661743164
# val_loss : 1.0985275506973267
# accuracy : 0.336848646402359
# val_accuracy : 0.35396039485931396