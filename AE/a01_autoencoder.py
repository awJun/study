"""
겐이 나오기 전까지 성능이 제일 좋았다고 한다.

_ : 메모리 저장을 안하겠다 라는 뜻 즉, 안불러오겠다.

앞 뒤가 똑같은 오토인코더 ???  따라불러~~ ;;
즉, input과 output이 똑같다 ~
들어간데이터와 나온데이터를 동일하게 출력하겠다.

inpout데이터에서 특성이 있는 부분을 추출해서 재 조합하겠다.

7이라는 녀석을 넣어서 필요없는 특성을 제거한 후 output에서 다시 7을 뽑아내는 작업이다.

784 -> 64(임의의 노드값) -> 784 / 즉! input과 output이 동일한 상태로 맞춰주면 노드가 줄어든 64 부분을 거치면서 필요없는 특성부분은 다 제거가 되고 output으로 동일한량의
output이 나오지만 중요한 특성만 남겨진 상태로 output으로 받을 수 있다.

activation = sigmoid가 통과됨ㄴ 0 ~ 1 사이로 된다 즉,! 스케일링이 된 상태가 된다. 그래서 sigmoid를 안하고 그냥 스케일링으로 해주면 된ㄷ.
이 작업을 해준 이유는 원본과 같은 사이즈로 맞추기 위해서 값을  0 ~ 1 사이로 제한을 거는 것임
sigmoid 전에는 아무 activation을 넣어도 무방하다.

오토인코더와 Gan에서 acc의 수치를 크게 신용하지마라! 즉, 조금 좋아진 거 같네 ? 아니네 ? 정도로 판단할 것 
acc말고 loss를 좀 더 신용하는 것이 좋다.


[프로젝트 정리본]
https://colab.research.google.com/drive/1EmkH1Px9SLxW12Sj2y87EyHxv-AZUrd9#scrollTo=di2fKBAQfuOY
"""

import numpy as np
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32")/255.    # 255.  : 부동소수점으로 출력하겟다.  / .astype("float32") 이것도 동일함 / 그래서 .astype("float32") 부분 없애도 255.해도 상관없음
x_test = x_test.reshape(10000, 784).astype("float32")/255.

from keras.models import Sequential, Model
from keras.layers import Dense, Input

imput_img = Input(shape=(784, ))
encoded = Dense(64, activation="relu")(imput_img)     # 원래 형태
# encoded = Dense(1064, activation="relu")(imput_img)   # 노드를 늘릴경우?  : 내생각: 노드가 input보다 커서 오토인코더가 전혀 되었다.
# encoded = Dense(16, activation="relu")(imput_img)       # 노드를 너무 줄일경우 : 특성을 너무 많이빼서 형태를 남아있지만 특성이 너무 많이 빠져있는 것을 볼 수 있었다.

decoded = Dense(784, activation="sigmoid")(encoded)   # 히든 레이어의 범위로 맞춰줬다.
# decoded = Dense(784, activation="relu")(encoded)      # 히든 레이어의 범위밖에 값이 나올수도있다.
# decoded = Dense(784, activation="linear")(encoded)    # 
# decoded = Dense(784, activation="tanh")(encoded)        # -1 ~ 0

autoencoder = Model(imput_img, decoded)

# autoencoder.summary()

autoencoder.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
# autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128,
                validation_split=0.2)   # 중지도 학습 즉, 지도학습도 비지도학습도 아닌 그 중간사이의 학습을 하는 것.
                                        # 중지도학습 : x_train으로 x_train을 훈련시킨다.  이거는
                                        # x_train는 x_train이야 즉, 필요없는 특성이 제거되는 과정을 하는 방법이다.
                                        # 약한 수치는 제거가되고 큰 수치가 살아남은 원리이다.  
                             
decoded_imgs = autoencoder.predict(x_test)


import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()













