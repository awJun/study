# 본인 사진으로 predict하시오!!
# d:/_study_data/_data/image/ 안에 넣고
# 데이터를 만들고 2로 보내기
# 평기지표 loss  accuracy 


import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255
    )

datasets = datagen.flow_from_directory(
    'D:\study_data\_data\image\men_women', # 이 경로의 이미지파일을 불러 수치화
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=9000,
    class_mode='binary', 
    # color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )

# print(datasets[0][0].shape)   
# print(datasets[0][1].shape)



np.save('D:/study_data/_save/_npy/men_women/keras47_04_x.npy', arr=datasets[0][0])
np.save('D:/study_data/_save/_npy/men_women/keras47_04_y.npy', arr=datasets[0][1])













































