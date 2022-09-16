# 본인 사진으로 predict하시오!!
# d:/_study_data/_data/image/ 안에 넣고
# 데이터를 만들고 2로 보내기
# 평기지표 loss  accuracy 


import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
datagen = ImageDataGenerator(
    rescale=1./255  # MinMax 스케일링과 같은 개념 
  )                 # 회전 축소 등으로 이미지에 여백이생겼을때 채우는 방법
    

datasets = datagen.flow_from_directory(
    'D:\study_data\_data\image\men_women/', # 이 경로의 이미지파일을 불러 수치화
    target_size=(100,100),# 크기들을 일정하게 맞춰준다.
    batch_size=10000,
    class_mode='binary', 
    # color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )


# dog = datagen.flow_from_directory(
#     'D:/study_data/_data/dog/', # 이 경로의 이미지파일을 불러 수치화  # 해당 경로에는 폴더가 있어야한다. 폴더안에 사진이 들어있어야폴더를 클래스로 인식한다.
#     target_size=(100,100),# 크기들을 일정하게 맞춰준다.
#     batch_size=1,
#     class_mode='binary', 
#     # color_mode='grayscale', #디폴트값은 컬러
#     shuffle=True,
#     )

x = datasets[0][0]  # x데이터
y = datasets[0][1]  # y데이터
# dog_test = dog[0][0]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)


np.save('D:/study_data/_save/_npy/men_women/keras47_04_x_train.npy', arr=x_train)
np.save('D:/study_data/_save/_npy/men_women//keras47_04_y_train.npy', arr=y_train)
np.save('D:/study_data/_save/_npy/men_women/keras47_04_x_test.npy', arr=x_test)
np.save('D:/study_data/_save/_npy/men_women/keras47_04_y_test.npy', arr=y_test)

# np.save("D:/study_data/_save/_npy/men_women/keras47_04_dog_test.npy",arr=dog_test)













































