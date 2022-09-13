"""
데이터에 노이즈를 씌우고 노이즈 없는 것으로 같이 훈련을 시킨다.

씌우면 0.0은 0.1 정도로 수치가 늘어난다

오토인코더는 해당도 자체가 흐릿해지는 단점이 있다.
하지만 겐에서 쓰면 해상도 문제는 없었다.


[해당 프로젝트 설명]
https://colab.research.google.com/drive/1ZQl3-PBeTR6z36vtmr98T3kEjjSqng8Z#scrollTo=uBXkYyfwdOCd
위에 프로젝트에서는 결과가 그림으로 안나옴.. ㅠ 그림은 이걸로 보자

"""


import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32")/255.    # 255.  : 부동소수점으로 출력하겟다.  / .astype("float32") 이것도 동일함 / 그래서 .astype("float32") 부분 없애도 255.해도 상관없음
x_test = x_test.reshape(10000, 784).astype("float32")/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)      # 0 ~1 범위인데 1 넘는 녀석을 컷 해줄것이다.

### 1이 넘는 녀석을 컷해준다. #################################
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
#############################################################

from keras.models import Sequential, Model
from keras.layers import Dense, Input

# Sequential를 함수로 만듬
def autoencoder(hidden_layers_size):
    model = Sequential()
    model.add(Dense(units=hidden_layers_size, input_shape=(784, ),
                    activation="relu"))
    model.add(Dense(units=784, activation="sigmoid"))
    return model

model = autoencoder(hidden_layers_size=154)    # 784의 95%의 성능은 154개이다 그래서 히든은 784개로 세팅했다.
# model = autoencoder(hidden_layers_size=331)    # 784의 99%의 성능은 331개이다 그래서 히든은 784개로 세팅했다.

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
random_images = random.sample(range(output.shape[0]), 5)

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



# PCA : 차원축소 약 95% 정도로 줄였을 때 성능이 좋게나온다
# 오토인코더를 사용할 때 hidden_layers_size를 pca로 어느정도 줄였을 때 성능이 좋은지 한 번 체크를 해보고 그 수치로 hidden_layers_size에 세팅하는게 좋다.

# PAC와 오토인코더는 둘 다 동일하게 차원을 축소하는 개념이다.

# enumerate : ??

























