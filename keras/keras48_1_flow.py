from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(   # ImageDataGenerator 이미지를 숫자화(수치화)
    rescale=1./255,   # 나는 1개의 이미지를 픽셀 255개로 나눌거야 스케일링한 데이터로 가져오겠다 ..? 
    horizontal_flip=True,
    vertical_flip=True,  # 반전시키겠냐 ? / true 네! 라는 뜻이라고함
    width_shift_range=0.1, # 가로 세로
    height_shift_range=0.1, # 상 하
    rotation_range=5, # 돌리겟다?
    zoom_range= 0.1, # 확대
    shear_range= 0.7, # 선생님 : 알아서 찾아 ~ ;;;  /  선생님 : 찌글찌글 ?? ;
    fill_mode='nearest'  
)

augument_size = 100

# print(x_train[0].shape)   # (28, 28)
# print(x_train[0].reshape(28*28).shape)  #  (784,)   / 스칼라가 784개 백터가 1개라는 뜻 ~
# print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1).shape)   # (100, 28, 28, 1)    <<-- x데이터로 사용
                                                                                        # (장수, 행, 열, 채널) 채널: 행 열이 2겹~3겹이 있다.  /  장수: 채널이 몇 장있는가
# print(np.zeros(augument_size))
# print(np.zeros(augument_size).shape)  # (100,)     <<--- y데이터로 사용

x_data = train_datagen.flow(   # 100장으로 증폭시켜줌
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1),
    np.zeros(augument_size),
    batch_size=augument_size,
    shuffle=True
).next()
# print(x_data)   
# <keras.preprocessing.image.NumpyArrayIterator object at 0x000001D9BC454E50>
print(x_data[0])  # <--batch만큼 나옴    # x와 y가 모두 포함

# print(x_data[0][0])  # <--batch만큼 나옴
# print(x_data[0][1])  # <--batch만큼 나옴

#==[ next()사용 안했을 때 ]=================================================
print(x_data[0][0].shape)  # <--batch만큼 나옴     # (100, 28, 28, 1)
print(x_data[0][1].shape)  # <--batch만큼 나옴     # (100,)
#==[ next()사용했을 때 ]=================================================
# print(x_data[0][0].shape)  # <--batch만큼 나옴   # (28, 28, 1)
# print(x_data[0][1].shape)  # <--batch만큼 나옴   # (28, 28, 1)
#========================================================================

# import matplotlib.pylab as plt
# plt.figure(figsize = (7,7))     # figsize(가로길이,세로길이)
# for i in range(49):
#     plt.subplot(7, 7, i+1)
#     plt.axis('off')
#     # plt.imshow(x_data[0][i], cmap='gray')     # <-- .next 사용
#     plt.imshow(x_data[0][0][i], cmap='gray')  # <-- .next 미사용
# plt.show()    









