import numpy as np
from keras.preprocessing.image import ImageDataGenerator


#1. 데이터

# train_datagen와 test_datagen는 x데이터이므로 스케일링해도 괜찮음!
#==[ train_datagen 사용준비 단계 ]=============================================================================
train_datagen = ImageDataGenerator(   # ImageDataGenerator 이미지를 숫자화(수치화)
    rescale=1./255,   # 나는 1개의 이미지를 픽셀 255개로 나눌거야 스케일링한 데이터로 가져오겠다 ..? 
    horizontal_flip=True,    # [긁음] 수평방향 뒤집기(true)
    width_shift_range=0.1  , # [긁음]  좌우로 이동
    height_shift_range=0.1,  # [긁음]  수직방향 이동
    rotation_range=5,        # [긁음]  원본 이미지를 회전 
    zoom_range= 1.2,         # [긁음]  확대/축소 
    shear_range= 0.7,        # [긁음]  원본 이미지를 변형
    fill_mode='nearest'      # [긁음]  이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
)
#=============================================================================================================
#==[ test_datagen 사용준비 단계 ]=============================================================================
test_datagen = ImageDataGenerator( 
    rescale=1./255   # 평가 데이터는 증폭시키면 안되므로 원래있던거 그대로 사용해야하기 때문에 이것만 해준다.
)
#=============================================================================================================

xy_train = train_datagen.flow_from_directory(   # directory : 폴더   / 즉! 폴더에서 가져오겠다! 라고 하는거임
   'D:/study_data/_data/image/cat_dog/training_set',   # 아까 만든 d드라이브에 데이터를 넣어놨는데 그걸 불러오는 것임! 
    target_size=(150, 150),    # <-- 100 x 100을 150으로하면 알아서 크기를 증폭하고 그 반대면 알아서 줄여서 사용한다. 즉! 사용자 맘대로 수치를 정해도됨
    batch_size=9000,
    class_mode='binary',   # 여기서는 정상, 비정상 2가지로 분류하므로 2진법에서 사용하는 binary를 선언!
    shuffle=True,
    # color_mode='grayscale'
)
xy_test = train_datagen.flow_from_directory(   # directory : 폴더   / 즉! 폴더에서 가져오겠다! 라고 하는거임
   'D:/study_data/_data/image/cat_dog/test_set',   # 아까 만든 d드라이브에 데이터를 넣어놨는데 그걸 불러오는 것임! 
    target_size=(150, 150), 
    batch_size=9000,
    class_mode='binary',   # 여기서는 정상, 비정상 2가지로 분류하므로 2진법에서 사용하는 binary를 선언!
    shuffle=True,
    # color_mode='grayscale'
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape)     # (8005, 100, 100, 1)
print(y_train.shape)     # (8005,)
print(x_test.shape)      # (2023, 100, 100, 1)
print(y_test.shape)      # (2023,)

# 증폭 사이즈
  
augment_size = 3995
batch_size = 1
randidx = np.random.randint(x_train.shape[0], size=augment_size)    # (5, 5)
                                                                    # np.random.randint는 랜덤하게 int를 뽑아냄
                                                                    # x_train.shape[0] = 60000
                                                                    # x_train.shape[1] = 28
                                                                    # x_train.shape[2] = 28                               
# print(x_train.shape[0])   # 5
# print(randidx)          
# print(np.min(randidx), np.max(randidx)) 
# print(type(randidx))   

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape)    # (5, 100, 100, 1)
print(y_augmented.shape)    # (5,)


#==[ 4차원 데이터이므로 생략 ]=============================================================
# x_train = x_train.reshape(5, 100, 100, 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# x_augmented = x_augmented.reshape(x_augmented.shape[0], 
#                                   x_augmented.shape[1], 
#                                   x_augmented.shape[2], 1)

x_augmented = train_datagen.flow(x_augmented, y_augmented,
                              batch_size=augment_size,
                              shuffle=False).next()[0]   # next()[0] => x만 사용하겠다는 의미
                                                            # shuffle=False이므로 label값 변환없이 들어감. 나중에 y값을 그대로 쓸 수 있음
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

#==[ x_augmented를 사용해서 증폭한 데이터를 2로 전달 ~]=====================================
np.save('d:/study_data/_save/_npy/cat_dog/keras49_06_train_x.npy', x_train)
np.save('d:/study_data/_save/_npy/cat_dog/keras49_06_train_y.npy', y_train)
np.save('d:/study_data/_save/_npy/cat_dog/keras49_06_test_x.npy', x_test)
np.save('d:/study_data/_save/_npy/cat_dog/keras49_06_test_y.npy', y_test)
#=========================================================================================
