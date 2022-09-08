import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import DirectoryIterator
from keras.layers import MaxPooling2D, Dropout
from sympy import Max
import tensorboard 

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale = 1./255,       
    horizontal_flip= True,  
    vertical_flip=True,     
    width_shift_range=0.1,  
    height_shift_range=0.1, 
    rotation_range=5,       
    zoom_range=1.2,         
    shear_range=0.7,        
    fill_mode='nearest',
    featurewise_center=True,
    featurewise_std_normalization=True,       
    )

test_datagen = ImageDataGenerator(
    rescale=1./255    
)

xy_train = train_datagen.flow_from_directory(
   'D:/study_data/_data/image/cat_dog/training_set', 
    target_size=(150, 150), 
    batch_size=8005,          
    class_mode='categorical',   
    color_mode='rgb', 
    shuffle=True,   
)

xy_test = test_datagen.flow_from_directory(
    'D:/study_data/_data/image/cat_dog/test_set',  
    target_size=(150, 150),
    batch_size=2023,
    class_mode='categorical',    
    color_mode='rgb',
    shuffle=True,
)

np.save('d:/study_data/_save/_npy/cat_dog/keras49_06_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/cat_dog/keras49_06_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/cat_dog/keras49_06_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/cat_dog/keras49_06_test_y.npy', arr=xy_test[0][1])