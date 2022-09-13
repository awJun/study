# [실습] 말하지 않아도 알아요~!! 시작해라..

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, UpSampling2D, Flatten

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape) 
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)   

### 1이 넘는 녀석을 컷해준다. #################################
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
#############################################################

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2, 2), strides=2, input_shape=(28, 28, 1), activation='relu'))
    # model.add(Flatten())
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=5, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.add(UpSampling2D(size=(2, 2), interpolation='nearest'))

    model.summary()
    return model

model = autoencoder(hidden_layer_size=154)    # 784의 95%의 성능은 154개이다 그래서 히든은 784개로 세팅했다.
# model = autoencoder(hidden_layer_size=331)    # 784의 99%의 성능은 331개이다 그래서 히든은 784개로 세팅했다.

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

model.fit(x_train_noised, x_train, epochs=10)
# 노이즈를 생성한 것은 x / 실제값인 y에 노이즈가 없는 x_train을 세팅한다.

output = model.predict(x_test_noised)


from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
    (ax11, ax12, ax13, ax14, ax15)) =\
    plt.subplots(3, 5, figsize=(20, 7))
    
# 이미지 다섯 개를 무작위로 고른다.
random_images = random = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap="gray")
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음(노이즈)을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap="gray")
    if i == 0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap="gray")
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()








