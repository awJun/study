"""
[VGG16 함수에서는 3개의 매개변수를 전달합니다.]   # https://subinium.github.io/Keras-5-2/

weights는 모델을 초기화할 가중치 체크포인트를 지정합니다.

include_top은 네트워크의 최상위 완전 연결 분류기를 포함할지 안할지를 지정합니다. 기본값은 ImageNet의 1,000개의 클래스에 대응되는 완전 연결 분류기를 포함합니다.
별도의 (강아지와 고양이 두 개의 클래스를 구분하는) 완전 연결 층을 추가하려고 하므로 이를 포함시키지 않습니다.

input_shape은 네트워크에 주입할 이미지 텐서의 크기입니다. 이 매개변수는 선택사항입니다. 이 값을 지정하지 않으면 네트워크가 어떤 크기의 입력도 처리할 수 있습니다.


 # CNN연산 : 3 x 3 (필터 크기) x 32 (#입력 채널) x 64(#출력 채널) + 64 = 18496 입니다.  / https://gaussian37.github.io/dl-keras-number-of-cnn-param/
 
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import VGG16

# model = VGG16()    # 디폴트 :  include_top=True / input_shape =(224, 224, 3)
vgg16 = VGG16(weights="imagenet", include_top=False,   # False하면 shape에러는 안나나 conn 형태 아래로 삭제되어서 13개가 남앗다 여기서 * 2를해서 26나옴   / w * b
              input_shape=(32, 32, 3))   # 나는 전이학습을 할 거야 하지만 32, 32, 3으로 변환해서 할거야


# vgg16.summary()
# vgg16.trainable=False  # 가중치를 동결시킨다.!!!
# vgg16.summary()

model = Sequential()
model.add(vgg16)   # 방법많음 함수형도 있고 좀 많아~ 나중에 필요할 때 찾아서 해도 문제없어
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))


# model.trainable = False   Truㄱ 평상지 


model.summary()         
                                    # Trainable:True  / VGG False /  model False
print(len(model.weights))            # 30 / 30 / 30    <-- len이므로 weights의 갯수임
print(len(model.trainable_weights))  # 30 /  4 /  0    #   w * b 




















