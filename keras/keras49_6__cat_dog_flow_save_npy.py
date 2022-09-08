import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale = 1./255,       # 스케일 조정
    horizontal_flip= True,  # 수평으로 반전
    vertical_flip=True,     # 수직으로 반전
    width_shift_range=0.1,  # 수평 이동 범위
    height_shift_range=0.1, # 수직 이동 범위
    rotation_range=5,       # 회전 범위
    zoom_range=1.2,         # 확대 범위
    shear_range=0.7,        # 기울기 범위
    fill_mode='nearest'     # 채우기 모드
    )

test_datagen = ImageDataGenerator(
    rescale=1./255    
)

xy_train = train_datagen.flow_from_directory(           # directory : 폴더   / 즉! 폴더에서 가져오겠다! 라고 하는거임
    'D:/study_data/_data/image/cat_dog/training_set',   # 아까 만든 d드라이브에 데이터를 넣어놨는데 그걸 불러오는 것임! 
    target_size=(150, 150),     # <-- 100 x 100을 150으로하면 알아서 크기를 증폭하고 그 반대면 알아서 줄여서 사용한다. 즉! 사용자 맘대로 수치를 정해도됨
    batch_size=8005,          
    class_mode='categorical',   # 여기서는 정상, 비정상 2가지로 분류하므로 2진법에서 사용하는 binary를 선언!
    color_mode='rgb',           # grayscale : 그레이로 출력  /   rgb : 컬러로 출력
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




