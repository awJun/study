from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D 

import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape)
print(np.unique(y_train, return_counts=True))

# 만들어보기
# acc 0.98 이상



print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    # (10000, 28, 28) (10000,)

print(x_train[0])  # 첫번째 x_train에 있는 값이 출력된다.
print(y_train[0])

import matplotlib.pyplot as plt
plt.imshow(x_train[5], 'gray')
plt.show()






model = Sequential()


#----------------------------------------------------------------------------
# 위 아래 같은 것                                  #
model.add(Conv2D(filters=64, kernel_size=(3,3),  # 출력 (N, 4, 4, 10)
                #  padding='same',   # padding은 원래 shape를 그대로 유지하고 싶을 때 사용한다.
                 input_shape=(28, 28, 1)))     #(batsh_size, row, columns, channels)
# padding은 현재 쉐이프를 다음 레이어에 그대로 사용하고 싶을 때 사용한다.

model.add(MaxPool2D())  #kernel_size=(3,3)로 잘랏을 때 자를 때 마다 해당 안에 최대값만 남긴다.

#  channels는 장수  / 1장 2장
model.add(Conv2D(32, (2, 2),  
                #  padding='valid',   # padding='valid' 디폴트 값 
                                    # 위에서 padding='same'을 하여 자동으로 들어감
                                    # 생략가능
                 activation='relu'))     # 출력(N 3, 3, 7)

# 이미지 데이터는 하나로 쫙~ 펼쳐서 연산을 한다.
#              v--- 데이터를 하나로 쫙 펼쳐줌
model.add(Flatten())  # (N, 63)   # 출력(N 3, 3, 7)  -->  3 * 3 * 7 = 27
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
#---------------------------------------------------------------------------



model.summary()



















































