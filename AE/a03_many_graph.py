"""
[핵심]



[해당 프로젝트 설명]
https://colab.research.google.com/drive/19mSa-dFD-dzvRiyMZwyLAtaOh7_8-YNl#scrollTo=JZrGQw9EyLXJ
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
    model.add(Dense(units=784, activation="sigmoid"))
    return model

model_01 = autoencoder(hidden_layers_size=1)
model_04 = autoencoder(hidden_layers_size=4)
model_16 = autoencoder(hidden_layers_size=16)
model_32 = autoencoder(hidden_layers_size=32)
model_64 = autoencoder(hidden_layers_size=64)
model_154 = autoencoder(hidden_layers_size=154)

### 노드의 갯수가 많아질수록 그림이 흐린상태에서 점점 더 선명해지는 것을 보기위해서 이 작업을 해봤습니다.
# fit 정의
print("============== node 1개 시작 ================")
model_01.compile(optimizer="adam", loss="binary_crossentropy")
model_01.fit(x_train, x_train, epochs=10)

print("============== node 4개 시작 ================")
model_04.compile(optimizer="adam", loss="binary_crossentropy")
model_04.fit(x_train, x_train, epochs=10)

print("============== node 16개 시작 ================")
model_16.compile(optimizer="adam", loss="binary_crossentropy")
model_16.fit(x_train, x_train, epochs=10)

print("============== node 32개 시작 ================")
model_32.compile(optimizer="adam", loss="binary_crossentropy")
model_32.fit(x_train, x_train, epochs=10)

print("============== node 64개 시작 ================")
model_64.compile(optimizer="adam", loss="binary_crossentropy")
model_64.fit(x_train, x_train, epochs=10)

print("============== node 154개 시작 ================")
model_154.compile(optimizer="adam", loss="binary_crossentropy")
model_154.fit(x_train, x_train, epochs=10)

# predict 정의
output_01 = model_01.predict(x_test)
output_04 = model_04.predict(x_test)
output_16 = model_16.predict(x_test)
output_32 = model_32.predict(x_test)
output_64 = model_64.predict(x_test)
output_154 = model_154.predict(x_test)

# 그림그리기axes
from matplotlib import pyplot as plt
import random

fig, axes = plt.subplots(7, 5, figsize=(15, 15))

random_imgs = random.sample(range(output_01.shape[0]), 5)
outputs = [x_train, output_01, output_04,
           output_16, output_32, output_64, output_154]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28, 28), cmap="gray")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()