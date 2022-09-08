from keras.applications import VGG16, VGG19
from keras.applications import ResNet50, ResNet50V2
from keras.applications import ResNetRS101, ResNet101V2, ResNet152, ResNet152V2
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.applications import InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2
from keras.applications import MobileNetV3Small, MobileNetV3Large
from keras.applications import NASNetLarge, NASNetMobile
from keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7   # 얘는 상당히 괜찮은 녀석이야 ~
from keras.applications import Xception
import time

# list = [VGG16, ...]

# model = VGG16()
# model = VGG19()
# #...

# model.trainable=False
# model.summary()

# print("==================================")
# print("모델명 : ", )
# print("전체 가중치 갯수 : ", len(model.weights))
# print("훈련 가능 가중치 갯수 : ", len(model.trainable))

####### 위에껄로 한번씩 다 돌리기 시작~ #######################################################

import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential,Model
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout, GlobalAveragePooling2D



#1.데이터
(x_train,y_train),(x_test,y_test)  = cifar10.load_data()



#2. 모델구성

models = [VGG16, VGG19,  ResNet50, ResNet50V2, ResNetRS101, ResNet101V2, ResNet152,
          ResNet152V2, DenseNet121, DenseNet169, DenseNet201, InceptionV3, InceptionResNetV2,
          MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large, NASNetLarge, NASNetMobile,
          EfficientNetB0, EfficientNetB1, EfficientNetB7, Xception]

for i in models:
    models_list = i

    model_1 = models_list(weights="imagenet", include_top=False,  
                input_shape=(32, 32, 3))
    model = Sequential()
    model.add(model_1) 
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))  
    model.add(Dense(64, activation='relu'))  
    model.add(Dense(10, activation="softmax")) 

    model_1.trainable=False   # 모델
    model.summary()

    print("==================================")
    print("모델명 : ", i.__name__ )
    print("전체 가중치 갯수 : ", len(model.weights))
    print("훈련 가능 가중치 갯수 : ", len(model.trainable_weights))

