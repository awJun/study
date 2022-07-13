# mnist 이미지 데이터셋
# 이미지 확인

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()   # 해당 데이터셋은 개발자가 알아서 train과 test가 나눠서 만들어주심

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)

print(x_train[0])  # 첫번째 x_train에 있는 값이 출력된다.
print(y_train[0])

import matplotlib.pyplot as plt
plt.imshow(x_train[5], 'gray')    # x_train 데이터중 5번째 인덱스의 그림을 출력해라 (그레이 색인 그림이므로 그레이 적음)
plt.show()

































