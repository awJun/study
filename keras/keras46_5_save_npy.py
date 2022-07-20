import numpy as np
from keras.preprocessing.image import ImageDataGenerator


#1. 데이터

# train_datagen와 test_datagen는 x데이터이므로 스케일링해도 괜찮음!
#==[ train_datagen 사용준비 단계 ]=============================================================================
train_datagen = ImageDataGenerator(   # ImageDataGenerator 이미지를 숫자화(수치화)
    rescale=1./255,   # 나는 1개의 이미지를 픽셀 255개로 나눌거야 스케일링한 데이터로 가져오겠다 ..?
    # [긁음] 밀린강도 범위내에서 임의로 원본 이미지를 변형 시킵니다. 수치는 시계반대방향으로 밀림강도를 radian으로 나타냅니다. 
#     horizontal_flip=True,
#     # [긁음] 수평방향 뒤집기(true)
#     vertical_flip=True,  # 반전시키겠냐 ? / true 네! 라는 뜻이라고함
#     # [긁음] 수직 방향 뒤집기(true)
#     width_shift_range=0.1, # 가로 세로
#     # [긁음]  지정된 수평방향 이동 범위 내에서 임의로 원본 이미지를 이동 시킵니다. 수치는 전체 넓이의 비율(실수)로 나타냅니다. ex> 0.1이면 전체 넓이 100의 10px 좌우로 이동
#     height_shift_range=0.1, # 상 하
#     # [긁음]  지정된 수직방향 이동 범위 내에서 임의로 원본 이미지를 이동 시킵니다. 수치는 전체 넓이의 비율(실수)로 나타냅니다
#     rotation_range=5, # 돌리겟다?
#     # [긁음]  지정된 각도 범위내에서 임의로 원본 이미지를 회전 시킵니다. 단위 도, 정수형 ex>rotaion_range=90
#     zoom_range= 1.2, # 확대
#     # [긁음]  지정된 확대/축소 범위내에 임의로 원본이미지를 확대/ 축소 합니다. (1 - 수치) ~ (1+ 수치) 사이의 범위로 확대 축소를 합니다
#     shear_range= 0.7, # 선생님 : 알아서 찾아 ~ ;;;  /  선생님 : 찌글찌글 ?? ;
#     # [긁음]  밀린강도 범위내에서 임의로 원본 이미지를 변형 시킵니다. 수치는 시계반대방향으로 밀림강도를 radian으로 나타냅니다. 
#     fill_mode='nearest'  
#     # [긁음]  # 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
)
#=============================================================================================================
#==[ test_datagen 사용준비 단계 ]=============================================================================
test_datagen = ImageDataGenerator( 
    rescale=1./255   # 평가 데이터는 증폭시키면 안되므로 원래있던거 그대로 사용해야하기 때문에 이것만 해준다.
)
#=============================================================================================================

xy_train = train_datagen.flow_from_directory(   # directory : 폴더   / 즉! 폴더에서 가져오겠다! 라고 하는거임
   'd:/study_data/_data/image/brain/train/',   # 아까 만든 d드라이브에 데이터를 넣어놨는데 그걸 불러오는 것임! 
    target_size=(150, 150),    # <-- 100 x 100을 150으로하면 알아서 크기를 증폭하고 그 반대면 알아서 줄여서 사용한다. 즉! 사용자 맘대로 수치를 정해도됨
    batch_size=500,
    class_mode='binary',   # 여기서는 정상, 비정상 2가지로 분류하므로 2진법에서 사용하는 binary를 선언!
    shuffle=True,
    color_mode='grayscale'
)
    # Found 160 images belonging to 2 classes.
# 데이터가 160개가 있고 그 안에 2개의 클래스가 있다  2개 클래스= 비정상(ad), 정상(normal) 
# 160개를 batch_size=5를 해서 5개씩 나눔  / 이것을 통해서 32개로 나눠짐 인덱스는 0부터이므로 아래에서는 [31]까지 사용가능함.
# [추가설명] 만약 batch_size=8이면 20개로 나눠지고 [19]까지 사용이 가능하다. !


xy_test = train_datagen.flow_from_directory(   # directory : 폴더   / 즉! 폴더에서 가져오겠다! 라고 하는거임
   'd:/study_data/_data/image/brain/test/',   # 아까 만든 d드라이브에 데이터를 넣어놨는데 그걸 불러오는 것임! 
    target_size=(150, 150),
    batch_size=500,
    class_mode='binary',   # 여기서는 정상, 비정상 2가지로 분류하므로 2진법에서 사용하는 binary를 선언!
    shuffle=True,
    color_mode='grayscale'
)
    # Found 120 images belonging to 2 classes.  
# 데이터가 120개가 있고 그 안에 2개의 클래스가 있다  2개 클래스= 비정상(ad), 정상(normal) 
# 160개를 batch_size=5를 해서 5개씩 나눔  / 이것을 통해서 24개로 나눠짐 인덱스는 0부터이므로 아래에서는 [23]까지 사용가능함.
# [추가설명] 만약 batch_size=8이면 15개로 나눠지고 [14]까지 사용이 가능하다. !


print(xy_train[0])     # <keras.preprocessing.image.DirectoryIterator object at 0x00000198C5295F70>
# array([1., 1., 0., 0., 1.]    batch_size가 5이므로 5개 나옴 ..  ???    
                                # 개는1 고양이는0으로 컴퓨터가 임의로 선택해서 해당 데이터에서 개와 고양이를 분리한 모습이 출력이된 모습같음

# 그냥 참고용
# print(xy_train[32]) 
# 즉! 160개중에 5개씩 잘려서 31개가 있는데 너는 32개를 요청해서 에러를 냈어 ! 라는뜻

# print(xy_train[31])
# print(xy_train[31][0].shape)   # (5, 150, 150, 3)  3은 칼라/  색을 따로 지정 안했으므로 흑백도 칼라로 인식 상관없음~ 흑백도 굳이 따지만 칼라색임 !
# [TMI] color_mode='grayscale'를 선언하면 흑백으로 인식해서 (5, 150, 150, 1) 3이아닌 1로 나온다. 
# 즉!  color_mode='grayscale'를 선언안하면 디폴트는 컬러이다 ~
 
# print(xy_train[31][1])   # [0. 0. 1. 0. 0.]

# print(xy_train[31][0].shape)   # (5, 150, 150, 1)
# xy_train[31][0]에서 [31]은 위에서 batch_size에서 잘라서 31번째에 해당하는 데이터 사진들을 가져온다는 뜻이고 
# xy_train[31][0]에서 [0]은 xy_train 파일에서의 순서 0은 젤 위 1은 젤 아래로 인식한다 즉! 0으로하면 ad(비정상)을 가져온다는 것이다.



#==[데이터의 자료형 사용하는 이유]=====================================
# print(type(xy_train))  #xy_train만 넣으면 이상하게 나오므로 type()으로 감싸서 xy_train의  자료형을 확인함. / type()는 자료형 확인용
# <class 'keras.preprocessing.image.DirectoryIterator'>

#==[데이터의 자료형 확인]=====================================
# print(type(xy_train[0]))  # <class 'tuple'>
# print(type(xy_train[1]))  # <class 'tuple'>

# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>
#============================================================
print(xy_train[0][0].shape, xy_train[0][1].shape)
print(xy_test[0][0].shape, xy_test[0][1].shape)


np.save('d:/study_data/_save/_npy/keras46_05_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras46_05_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras46_05_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/keras46_05_test_y.npy', arr=xy_test[0][1])

# 현재 (5,200,200,1)라는 데이터가 32덩어리 ~

#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(100, 100, 1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])   # 
model.fit(xy_train[0][0], xy_train[0][1])  # 현재 x, y데이터 한덩어리씩 있으므로 이렇게함   해당은 배치사이즈를 최대로 할 경우 가능하다.

# 위에 문제를 해결하기위해 fit_generator를 사용
hist = model.fit_generator(xy_train, epochs=30,  #  그냥 데이터 통채로 넣으면된다
                   validation_data=xy_test, 
                   validation_steps=4,
                   #[중요!] validation_data=xy_test, validation_steps=4 를 사용해서 아래에서 accuracy = hist.history['accuracy']를 사용하기
                          # 위해서 선언함 / 선언해야지 metrics=['accuracy']와 만나서 아래에서 결과를 도출해줌 !
                  
                   steps_per_epoch=33,   # batch_size 대신 이걸 사용해야함   /  1 epoch당 훈련하는 양을 조절하는 것임!
                    # 범위는 데이터셋을 batch사이즈로 나눈거 즉!  -->  전체데이터/batch = 160/5 = 32 [정정]
                      # 지금보니까 그냥 제한 범위없음 마음대로 할 것 근데 통상적으로 위에처럼 사용한다고함.
                    # [중요] fit_generator에서는 batch_size를 사용할 수 없다 ~
                    )  

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])   # .history를 사용해서 로스값이 여러개가 출력이 되지만 그 중에서 제일 마지막것을 가져와서 보기위해 [-1를]사용 출력된 로스값 데이터들은
                                  # 리스트 형태로 담기므로 인덱스 번호가 먹는다 !
print('val_accuracy : ', val_accuracy[-1])
print('accuracy : ', accuracy[-1])
print('val_loss : ', val_loss[-1])

# loss :  0.6927464604377747
# val_accuracy :  0.550000011920929
# accuracy :  0.5515151619911194
# val_loss :  0.6897107362747192


# 그림그려!!! 

import matplotlib.pyplot as plt

plt.figure(figsize = (9,6))     # figsize(가로길이,세로길이)
plt.plot(hist.history['accuracy'], marker='.', color='green', label='accuracy')           # label='loss' 해당 선 이름
plt.plot(hist.history['val_accuracy'], marker='.', color='yellow', label='val_accuracy')  # marker='.' 점으로 찍겟다
plt.plot(hist.history['loss'], marker='.', color='red', label='loss')  # marker='.' 점으로 찍겟다
plt.plot(hist.history['val_loss'], marker='.', color='blue', label='val_loss')  # marker='.' 점으로 찍겟다
plt.grid()                        # plt.grid(True)    # grid: 그리다
plt.title('ha_ha_ha')                # title의 이름을 asaql로 하겠다
plt.ylabel('loss')                # y라벨의 이름을 loss로 하겠다
plt.xlabel('epochs')              # x라벨의 이름을 epochs로 하겠다
plt.legend(loc = 'upper right')   # 그래프가 없는쪽에 알아서 해준다 굳이 명시할 필요 없음
# plt.legend()   # 그래프가 없는쪽에 알아서 해준다 굳이 명시를 안 할 경우 사용법
plt.show()    # 그래프를 보여줘라

