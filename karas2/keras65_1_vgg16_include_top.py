"""
[핵심]
VGG는 전이학습 모델이다.

전이학습 모델은 프로그램에서 처음으로 실행 모델들은 아래처럼 한 번씩 다운받고 시작한다 / 즉, 다음에 사용할 때는 다운 안받아도 된다.
379011072/553467096 [===================>..........] - ETA: 12s 

"""
# 19개 레이어 갯수에서 에서 

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.applications import VGG16

# model = VGG16()    # 디폴트 :  include_top=False / input_shape =(224, 224, 3)
model = VGG16(weights="imagenet", include_top=False,   # False하면 shape에러는 안나나 conn 형태 아래로 삭제되어서 13개가 남앗다 여기서 * 2를해서 26나옴   / w * b
              input_shape=(32, 32, 3))   # 나는 전이학습을 할 거야 하지만 32, 32, 3으로 변환해서 할거야

model.summary()

print(len(model.weights))   # 32   
print(len(model.trainable_weights))  # 32  모델에 훈련된 웨이트값


################ include_top = True #######################################
#1. FC layer 원래꺼 그대로 쓴다.
#2. input_shape=(244, 244, 3) 고정값 바꿀 수 없다.


#  input_1 (InputLayer)        [(None, 224, 224, 3)]     0

#  block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
#..........................................................................
#  flatten (Flatten)           (None, 25088)             0

#  fc1 (Dense)                 (None, 4096)              102764544

#  fc2 (Dense)                 (None, 4096)              16781312

#  predictions (Dense)         (None, 1000)              4097000

# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________
# 32
# 32


################ include_top = False #######################################
#1. FC layer 원래꺼 삭제 -> 나는야 커스터마이징을 할거다!!!
#2. input_shape=(32, 32, 3) 고정값 바꿀 수 있다. - 커스터마이징을 할거야!!!

#  input_1 (InputLayer)        [(None, 32, 32, 3)]       0
#  block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792
# ....................................................
# .......................................... 플래튼 하단 실종!!! 두그둥!!!
# 풀리커넥티드레이너 하단이 아디오스 하는거야!!!
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________
# 26
# 26







