import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

 
x_train = np.load('d:/study_data/_save/_npy/brain/keras49_05_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/brain/keras49_05_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/brain/keras49_05_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/brain/keras49_05_test_y.npy')

print(x_train.shape) # (10, 100, 100, 1)
print(y_train.shape) # (10,)
print(x_test.shape)  # (5, 100, 100, 1)
print(y_test.shape)  # (5,)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten
from sklearn.metrics import r2_score,accuracy_score
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape = (100,100,1), activation='relu'))
model.add(Conv2D(10, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

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

# loss : 0.014676118269562721
# val_loss : 9.870941162109375
# accuracy : 1.0
# val_accuracy : 0.0




