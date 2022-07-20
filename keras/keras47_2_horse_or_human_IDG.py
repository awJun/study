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
print(datasets[0][0].shape)   # (1027, 150, 150, 3)
print(datasets[0][1].shape)   # (1027,)

# np.save('D:/study_data/_save/_npy/horse_or_human/keras47_2_x_data.npy', arr=datasets[0][0])
# np.save('D:/study_data/_save/_npy/horse_or_human/keras47_2_y_data.npy', arr=datasets[0][1])









































