"""
해당내용 다읽어보기
"""

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten # Flatten: 평평한





       # Conv2D 이미지 

model = Sequential()
# model.add(Dense(units=10, input_shape=(3,)))   
#                # (batch_size, input_dim)
 
# # 위 아래 같은 것  

# model.add(Conv2D(filters=10, kernel_size=(2, 2),  # 출력 (N, 4, 4, 10)
#                  input_shape=(5, 5, 1)))     #(batsh_size, row, columns, channels)
# #  1은 장수  / 1장 2장

# model.add(Conv2D(7, (2, 2), activation='relu'))     # 출력(N 3, 3, 7)

#----------------------------------------------------------------------------
# 위 아래 같은 것                                  #
model.add(Conv2D(filters=10, kernel_size=(3,3),  # 출력 (N, 4, 4, 10)
                 input_shape=(8, 8, 1)))     #(batsh_size, row, columns, channels)


#  channels는 장수  / 1장 2장
model.add(Conv2D(7, (2, 2), activation='relu'))     # 출력(N 3, 3, 7)
model.add(Flatten())  # (N, 63)   # 출력(N 3, 3, 7)  -->  3 * 3 * 7 = 27
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
#----------------------------------------------------------------------------


# CNN 계산:   (kernel_size * channels + bias) * filters = summary Param 갯수

#input_shape의 channels-v         v-- input_shape의 channels
  # kernel_size -> (3 * 1) * (3 * 1) + 1 = 10 * 10 = 100  <-- summary Param 갯수
#kernel_size의 행-^         ^--kernel_size의 열   





# Dense 계산:  (input_dim + bias) * units = summary Param 갯수

# model.summary()
# Model: "sequential"
# _________________________________________________________________   
# Layer (type)                 Output Shape              Param #              v-- [연산] model.add(Conv2D(filters=10, kernel_size=(3,3), 
# =================================================================                                      input_shape=(8, 8, 1)))
# conv2d (Conv2D)              (None, 6, 6, 10)          100                (((3 * 3) * 1) + 1) * 10 =100   
# _________________________________________________________________   연산이후  [3x3]  -->  [2x2]
# conv2d_1 (Conv2D)            (None, 5, 5, 7)           287                (((2 * 2) * 10) + 1) * 7 = 287
# _________________________________________________________________   연산이후  [2x2]  -->  [1x1]
# flatten (Flatten)            (None, 175)               0               
# _________________________________________________________________           ^-- [연산] model.add(Conv2D(7, (2, 2), activation='relu'))
# dense (Dense)                (None, 32)                5632          데이터를 1x1까지 자른 후 flatten (Flatten) 이후로 기존처럼 연산 시작함 
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                1056
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                330
# =================================================================
# Total params: 7,405
# Trainable params: 7,405
# Non-trainable params: 0
# _________________________________________________________________























