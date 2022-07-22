import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

 
x_train = np.load('d:/study_data/_save/_npy/cat_dog/keras47_01_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/cat_dog/keras47_01_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/cat_dog/keras47_01_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/cat_dog/keras47_01_test_y.npy')

print(x_train.shape) # (8005, 150, 150, 3)
print(y_train.shape) # (8005,)
print(x_test.shape)  # (2023, 150, 150, 3)
print(y_test.shape)  # (2023,)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten
from sklearn.metrics import r2_score,accuracy_score
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape = (150,150,3), activation='relu'))
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
hist = model.fit(x_train, y_train, epochs=10, 
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

# loss : 0.002706121187657118
# val_loss : 2.0496702194213867
# accuracy : 0.999687671661377
# val_accuracy : 0.6770768165588379


#그래프로 비교
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.plot(hist.history['accuracy'], marker='.', c='orange', label='accuracy')
plt.plot(hist.history['val_accuracy'], marker='.', c='green', label='val_accuracy')
plt.grid()    
plt.title('로스값과 검증로스값')    
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()   
plt.show()













