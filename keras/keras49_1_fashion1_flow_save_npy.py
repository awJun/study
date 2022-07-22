# 넘파이에서 불러와서 모델 구성
# 성능 비교
from click import argument
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# fashion_mnist는 이미 수치화된 데이터이므로 스케일링은 차리 min max로 하거나
# 아예 스케일링을 안하는게 성능 좋은 거 같음.

train_datagen = ImageDataGenerator(
    
    # rescale=1./255,
    # horizontal_flip=True,
    # # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=0.1,
    # # shear_range=0.7,
    fill_mode='nearest'
)

train_datagen2 = ImageDataGenerator(
    rescale=1./255  
)

# 증폭 사이즈
augment_size = 40000
batch_size = 1    # 그냥 아래에서 사용하려고 임의로 숫자를 넣은거임 0은 에러나는 거 같음 0빼고 넣자 ~
randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0])    # 60000
print(x_train.shape[1])    # 28
print(x_train.shape[2])    # 28

print(x_train.shape)      # (60000, 28, 28)
# print(randidx)          # [31720 43951 44299 ... 22547 15575 47042]
# print(np.min(randidx), np.max(randidx)) # 3 59999
print(type(randidx))      # <class 'numpy.ndarray'>

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)    # (40000, 28, 28)
print(y_augmented.shape)    # (40000,)


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augmented = x_augmented.reshape(x_augmented.shape[0], 
                                  x_augmented.shape[1], 
                                  x_augmented.shape[2], 1)

x_augmented = train_datagen.flow(x_augmented, y_augmented,
                              batch_size=augment_size,
                              shuffle=False).next()[0] # x_augmented만 사용하려고 [0]선언

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

xy_train = train_datagen2.flow(x_train, y_train,
                              batch_size=batch_size,
                              shuffle=False)

xy_test = train_datagen2.flow(x_test, y_test,
                              batch_size=batch_size,
                              shuffle=False)

#==[ x_augmented를 사용해서 증폭한 데이터를 2로 전달 ~]=====================================
np.save('d:/study_data/_save/_npy/fashion_mnist/keras49_01_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/fashion_mnist/keras49_01_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/fashion_mnist/keras49_01_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/fashion_mnist/keras49_01_test_y.npy', arr=xy_test[0][1])
#=========================================================================================









































