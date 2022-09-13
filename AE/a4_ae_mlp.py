"""
[핵심]
레이어를 다층으로 구성해서 단층일때와 성능을 비교해볼 것

[결과]
레이어층이 단층인 상황과 다층인 상황에서 비교를 해봤을 때 두 개의 차이가 거의 없다.


[해당 프로젝트 설명]
https://colab.research.google.com/drive/1G4n9W9z_Z6YfUe_qQj6rrrW1-p6SXQQ2#scrollTo=bn2IsYVm1UA-
"""



import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32")/255.    # 255.  : 부동소수점으로 출력하겟다.  / .astype("float32") 이것도 동일함 / 그래서 .astype("float32") 부분 없애도 255.해도 상관없음
x_test = x_test.reshape(10000, 784).astype("float32")/255.

from keras.models import Sequential, Model
from keras.layers import Dense, Input

# Sequential를 함수로 만듬
def autoencoder(hidden_layers_size):
    model = Sequential()
    model.add(Dense(units=hidden_layers_size, input_shape=(784, ),
                    activation="relu"))
    model.add(Dense(units=500, activation="relu"))
    model.add(Dense(units=600, activation="relu"))
    model.add(Dense(units=784, activation="sigmoid"))
    return model

model = autoencoder(hidden_layers_size=154)    # 784의 95%의 성능은 154개이다 그래서 히든은 784개로 세팅했다.
# model = autoencoder(hidden_layers_size=331)    # 784의 99%의 성능은 331개이다 그래서 히든은 784개로 세팅했다.

model.compile(optimizer="adam", loss="binary_crossentropy")

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)


from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize=(20, 7))
    
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

for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap="gray")
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()



# PCA : 차원축소 약 95% 정도로 줄였을 때 성능이 좋게나오는 거 같다.
# 오토인코더를 사용할 때 hidden_layers_size를 pca로 어느정도 줄였을 때 성능이 좋은지 한 번 체크를 해보고 그 수치로 hidden_layers_size에 세팅하는게 좋다.

# PAC와 오토인코더는 둘 다 동일하게 차원을 축소하는 개념이다.

# enumerate : ??





