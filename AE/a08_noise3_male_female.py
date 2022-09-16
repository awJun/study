# [실습] keras47_4 남자 여자에 noise를 넣어서
# predict 첫 번째 : 기미 주근깨 여드름 제거!!!
# 랜덤하게 5개 정도 원본/수정본 빼고

# predict 두 번째 : 본인 사진넣어서 빼 !!!  /  원본 수정본
# 랜덤안하면 노이즈가 안퍼짐

import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense, Conv2D, UpSampling2D, Flatten, MaxPooling2D

x_train = np.load('D:/study_data/_save/_npy/men_women/keras47_04_x_train.npy')
x_test = np.load('D:/study_data/_save/_npy/men_women/keras47_04_x_test.npy')
dog_test = np.load('D:/study_data/_save/_npy/men_women/keras47_04_dog_test.npy')
# ,allow_pickle=True

print(dog_test)
print(dog_test.shape)




x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape) 
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)   
dog_noised = dog_test + np.random.normal(0, 0.1, size=(1, 100, 100, 3))   


x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
dog_noised = np.clip(dog_noised, a_min=0, a_max=1)


def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2, 2), padding="same", input_shape=(100, 100, 3), activation='relu'))
    model.add(Conv2D(50, (2, 2), activation="relu", padding="same"))
    model.add(MaxPooling2D())
    # model.add(Flatten())   # 이거랑 UpSampling2D랑 같이 사용하면 ValueError 에러발생함..
    model.add(UpSampling2D(size=(2, 2), interpolation='nearest'))
    model.add(Dense(units=3, activation='sigmoid'))

    model.summary()
    return model

model = autoencoder(hidden_layer_size=154)    # 784의 95%의 성능은 154개이다 그래서 히든은 784개로 세팅했다.
# model = autoencoder(hidden_layer_size=331)    # 784의 99%의 성능은 331개이다 그래서 히든은 784개로 세팅했다.


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

model.fit(x_train_noised, x_train, epochs=1)
# 노이즈를 생성한 것은 x / 실제값인 y에 노이즈가 없는 x_train을 세팅한다.

output = model.predict(x_test_noised)
dog_output = model.predict(dog_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5, dog1), (ax6, ax7, ax8, ax9, ax10, dog2),
    (ax11, ax12, ax13, ax14, ax15, dog3)) =\
    plt.subplots(3, 6, figsize=(20, 7))
    
# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(x_test.shape[0]), 5)
print(random_images)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]], cmap="gray")
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
dog1.imshow(dog_test.reshape(100, 100, 3), cmap="gray")


# 잡음(노이즈)을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(100, 100, 3), cmap="gray")
    if i == 0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
dog2.imshow(dog_noised.reshape(100, 100, 3), cmap="gray")


for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(100, 100, 3), cmap="gray")
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
dog3.imshow(dog_output, cmap="gray")


plt.tight_layout()
plt.show()

