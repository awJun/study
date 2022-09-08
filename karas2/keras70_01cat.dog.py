import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

#=[ 증폭된 데이터 불러옴 ]=======================================================
# keras49_6__cat_dog_flow_save_npy.py

x_train = np.load('d:/study_data/_save/_npy/cat_dog/keras49_06_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/cat_dog/keras49_06_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/cat_dog/keras49_06_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/cat_dog/keras49_06_test_y.npy')
#==============================================================================

# print(x_train.shape) # (12000, 150, 150, 3)
# print(y_train.shape) # (12000,)
# print(x_test.shape)  # (2023, 150, 150, 3)
# print(y_test.shape)  # (2023,)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten
from sklearn.metrics import r2_score,accuracy_score
model = Sequential()

model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.summary()

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mse','accuracy'])

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
print('val_accuracy :', val_accuracy[-1])


# loss : 0.3074779808521271
# val_loss : 0.697265625
# accuracy : 0.8462499976158142
# val_accuracy : 0.5099999904632568