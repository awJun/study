import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# train_datagen와 test_datagen는 x데이터이므로 스케일링해도 괜찮음!
#==[ train_datagen 사용준비 단계 ]=============================================================================
train_datagen = ImageDataGenerator(   # ImageDataGenerator 이미지를 숫자화(수치화)
    rescale=1./255,   # 나는 1개의 이미지를 픽셀 255개로 나눌거야 스케일링한 데이터로 가져오겠다 ..?
    # [긁음] 밀린강도 범위내에서 임의로 원본 이미지를 변형 시킵니다. 수치는 시계반대방향으로 밀림강도를 radian으로 나타냅니다. 
    horizontal_flip=True,
    # [긁음] 수평방향 뒤집기(true)
    vertical_flip=True,  # 반전시키겠냐 ? / true 네! 라는 뜻이라고함
    # [긁음] 수직 방향 뒤집기(true)
    width_shift_range=0.1, # 가로 세로
    # [긁음]  지정된 수평방향 이동 범위 내에서 임의로 원본 이미지를 이동 시킵니다. 수치는 전체 넓이의 비율(실수)로 나타냅니다. ex> 0.1이면 전체 넓이 100의 10px 좌우로 이동
    height_shift_range=0.1, # 상 하
    # [긁음]  지정된 수직방향 이동 범위 내에서 임의로 원본 이미지를 이동 시킵니다. 수치는 전체 넓이의 비율(실수)로 나타냅니다
    rotation_range=5, # 돌리겟다?
    # [긁음]  지정된 각도 범위내에서 임의로 원본 이미지를 회전 시킵니다. 단위 도, 정수형 ex>rotaion_range=90
    zoom_range= 1.2, # 확대
    # [긁음]  지정된 확대/축소 범위내에 임의로 원본이미지를 확대/ 축소 합니다. (1 - 수치) ~ (1+ 수치) 사이의 범위로 확대 축소를 합니다
    shear_range= 0.7, # 선생님 : 알아서 찾아 ~ ;;;  /  선생님 : 찌글찌글 ?? ;
    # [긁음]  밀린강도 범위내에서 임의로 원본 이미지를 변형 시킵니다. 수치는 시계반대방향으로 밀림강도를 radian으로 나타냅니다. 
    fill_mode='nearest'  
    # [긁음]  # 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
)
#=============================================================================================================
#==[ test_datagen 사용준비 단계 ]=============================================================================
test_datagen = ImageDataGenerator( 
    rescale=1./255   # 평가 데이터는 증폭시키면 안되므로 원래있던거 그대로 사용해야하기 때문에 이것만 해준다.
)
#=============================================================================================================

xy_train = train_datagen.flow_from_directory(   # directory : 폴더   / 즉! 폴더에서 가져오겠다! 라고 하는거임
   'd:/_data/image/brain/train/',   # 아까 만든 d드라이브에 데이터를 넣어놨는데 그걸 불러오는 것임! 
    target_size=(150, 150),    # <-- 100 x 100을 150으로하면 알아서 크기를 증폭하고 그 반대면 알아서 줄여서 사용한다. 즉! 사용자 맘대로 수치를 정해도됨
    batch_size=5,
    class_mode='binary',   # 여기서는 정상, 비정상 2가지로 분류하므로 2진법에서 사용하는 binary를 선언!
    shuffle=True,
    color_mode='grayscale'
)
    # Found 160 images belonging to 2 classes.
# 데이터가 160개가 있고 그 안에 2개의 클래스가 있다  2개 클래스= 비정상(ad), 정상(normal) 
# 160개를 batch_size=5를 해서 5개씩 나눔  / 이것을 통해서 32개로 나눠짐 인덱스는 0부터이므로 아래에서는 [31]까지 사용가능함.
# [추가설명] 만약 batch_size=8이면 20개로 나눠지고 [19]까지 사용이 가능하다. !


xy_test = train_datagen.flow_from_directory(   # directory : 폴더   / 즉! 폴더에서 가져오겠다! 라고 하는거임
   'd:/_data/image/brain/test/',   # 아까 만든 d드라이브에 데이터를 넣어놨는데 그걸 불러오는 것임! 
    target_size=(150, 150), 
    batch_size=5,
    class_mode='binary',   # 여기서는 정상, 비정상 2가지로 분류하므로 2진법에서 사용하는 binary를 선언!
    shuffle=True,
    color_mode='grayscale'
)
    # Found 120 images belonging to 2 classes.
# 데이터가 120개가 있고 그 안에 2개의 클래스가 있다  2개 클래스= 비정상(ad), 정상(normal) 
# 160개를 batch_size=5를 해서 5개씩 나눔  / 이것을 통해서 24개로 나눠짐 인덱스는 0부터이므로 아래에서는 [23]까지 사용가능함.
# [추가설명] 만약 batch_size=8이면 15개로 나눠지고 [14]까지 사용이 가능하다. !


# print(xy_train[0])     # <keras.preprocessing.image.DirectoryIterator object at 0x00000198C5295F70>
# array([1., 1., 0., 0., 1.]    batch_size가 5이므로 5개 나옴 ..  ???

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














