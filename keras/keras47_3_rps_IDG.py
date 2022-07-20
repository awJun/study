# 데이터를 만들고 2로 보내기
# 평기지표 loss  accuracy 

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255)

test_datagen = ImageDataGenerator(
    rescale=1./255) 

datasets = train_datagen.flow_from_directory(
    'D:/study_data/_data/image/rps/rps',   # <-- horse와 human이 동시에 담긴 폴더를 불러옴
    target_size=(150, 150), # 불러온 사진의 크기를 해당 크기로 일정하게 맞춰준다.
    batch_size=9000,
    class_mode='categorical',
    shuffle=True
)  # Found 1027 images belonging to 2 classes.
print(datasets[0][0].shape)   # (2520, 150, 150, 3)
print(datasets[0][1].shape)   # (2520,)

np.save('D:/study_data/_save/_npy/rps/keras47_3_x_data.npy', arr=datasets[0][0])
np.save('D:/study_data/_save/_npy/rps/keras47_3_y_data.npy', arr=datasets[0][1])














































