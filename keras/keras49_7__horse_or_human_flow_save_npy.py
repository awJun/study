# 데이터를 만들고 2로 보내기
# 평기지표 loss  accuracy 

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255)

test_datagen = ImageDataGenerator(
    rescale=1./255)  #  그냥 개발자가 이렇게 만들었는데 이렇게 하면 min max 스케일러를 적용 해준다.

datasets = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/horse-or-human/horse-or-human/',   # <-- horse와 human이 동시에 담긴 폴더를 불러옴
    target_size=(150, 150), # 불러온 사진의 크기를 해당 크기로 일정하게 맞춰준다.
    batch_size=9000,
    class_mode='binary',
    shuffle=True
)  # Found 1027 images belonging to 2 classes.
# print(datasets[0][0].shape)   # (1027, 150, 150, 3)
# print(datasets[0][1].shape)   # (1027,)

x = datasets[0][0]
y = datasets[0][1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

print(x_train.shape)    # (821, 150, 150, 3) 
print(x_test.shape)     # (206, 150, 150, 3)
print(y_train.shape)    # (821,)
print(y_test.shape)     # (206,)

# 증폭 사이즈
  
augment_size = 179
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
print(x_augmented.shape)    # (179, 150, 150, 3)
print(y_augmented.shape)    # (179,)


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

print(x_train.shape)   # (1000, 150, 150, 3)
print(y_train.shape)   # (1000,)

np.save('D:/study_data/_save/_npy/horse_or_human/keras49_7_x_train_data.npy', x_train)
np.save('D:/study_data/_save/_npy/horse_or_human/keras49_7_y_train_data.npy', y_train)
np.save('D:/study_data/_save/_npy/horse_or_human/keras49_7_x_test_data.npy', x_test)
np.save('D:/study_data/_save/_npy/horse_or_human/keras49_7_y_test_data.npy', y_test)









































