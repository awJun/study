from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D 
# Flatten: 평평한


       # Conv2D 이미지 
model = Sequential()
# model.add(Dense(units=10, input_shape=(3,)))      # input_shape=(10, 10, 3)))  <-- 일단 배제
#                # (batch_size, input_dim)
 
# # 위 아래 같은 것                                  #
# model.add(Conv2D(filters=10, kernel_size=(2, 2),  # 출력 (N, 4, 4, 10)
#                  input_shape=(5, 5, 1)))     #(batsh_size, row, columns, channels)
# #  1은 장수  / 1장 2장
# model.add(Conv2D(7, (2, 2), activation='relu'))     # 출력(N 3, 3, 7)

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










# CNN 계산:   (kernel_size * channels + bias) * filters = summary Param 갯수

#input_shape의 channels-v         v-- input_shape의 channels
  # kernel_size -> (3 * 1) * (3 * 1) + 1 = 10 * 10 = 100  <-- summary Param 갯수
#kernel_size의 행-^         ^--kernel_size의 열   


# Dense 계산:  (input_dim + bias) * units = summary Param 갯수

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 6, 6, 10)          100
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 5, 5, 7)           287
# _________________________________________________________________
# flatten (Flatten)            (None, 175)               0
# _________________________________________________________________
# dense (Dense)                (None, 32)                5632
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                330
# =================================================================
# Total params: 7,405
# Trainable params: 7,405
# Non-trainable params: 0
# _________________________________________________________________























