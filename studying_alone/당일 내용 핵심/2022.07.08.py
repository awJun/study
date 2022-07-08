
#===============================================================================================#
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D

       # Conv2D 이미지관련 
       
       
model = Sequential()
#[2차원]
# model.add(Dense(units=10, input_shape=(3,)))      # input_shape=(10, 10, 3)))  <-- 일단 배제
             # (batch_size, input_dim)      batch_size <-- 행  /  input_dim <-- 열
             # (10, input_dim=3)            10  <-- 행   /  input_dim <-- d열
 #           위에 3개다 같은 뜻이다. 
             # n, INPUT_dim
             # N, output
# Dense 계산:  (input_dim + bias) * units = summary Param 갯수
 
 #===============================================================================================#
#[3차원]   # Conv2D 관련링크  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
model.add(Conv2D(filfers=10, kernel_size=(2, 2), strides=1,
                 input_shape=(5, 5, 1)))     #(batsh_size, row, columns, channels)
                                       # input n, r, c, c  
                                      # output n, R, c, f
     # filfers=10  <-- 행   /  kernel_size=(2, 2)  /   input_shape=( , , 1)))   3개는 units와 같다.
     # input_shape=(5, 5,  )))   <-- input_dim
     # batsh_size  <-- 행    row  <-- 행  /  columns  <-- 열   channels  <-- 열 
     # 행2개 열2개 행열연산 가능           
                
                 # 5 x 5 이미지 흑백  [1 = 흑백   3 = 컬러]
                 # kernel_size=(2, 2)  2 x 2로 잘랐다 / 자르고 싶은 만큼 수정하면됨.
                 #  input_shape=(5, 5, 1))) 
                # 두 개가 하나의 레이어 연산 2
         
model.add(Conv2D(7, (2,2), activation='relu')) 
            # 7 == filfers=7  /   (2,2) == kernel_size=(2, 2)
            # 'relu'는 이미지에 특히 좋음 이미지는 음수가 없고 0~ 255 범위이기 때문이다.
model.summary()
# CNN 계산:   (kernel_size * channels + bias) * filters = summary Param 갯수
#===============================================================================================#

##### [ Dropout ] #####



# Dropout을 통해서 훈련중 노드 중간 중간 이빨을 빼듯이 지워버리지만 평가, 예측에서는
# 빠진 데이터를 사용하지 않고 전체 노드가 전부 적용된다.

from tensorflow.python.keras.layers import Dense, Input, Dropout #(데이터의 노드를 중간중간 날려줌)
                                                 # 데이터가 많을수록 성능 좋음.
                                                 
# [ Sequential version ]
 #2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dropout(0.3))      # 30%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))       # 20%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
#-------------------------------------------------------------------------------------#
# [ hamsu version ]
 #2. 모델구성
input1 = Input(shape=(3,))
dense1 = Dense(10)(input1)
dense2 = Dense(15, activation'relu')(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs=input1, outputs=output1)
#===============================================================================================#





























