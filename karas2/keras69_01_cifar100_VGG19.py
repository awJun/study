# shape 오류인거는 내용 명시가고 추가 모델 맹그러



import numpy as np
from tensorflow.keras.datasets import mnist, cifar100
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout, GlobalAveragePooling2D
import keras
import tensorflow as tf 
from keras.applications import VGG19
from tensorflow.keras.preprocessing import image
from keras.applications.resnet import preprocess_input, decode_predictions   # 케라스에서 지원하는 것은 이것으로 스케일링 할 수 있게 지원해줌 근데 이거말고 정네처럼 스케일링 해도 괜찮음



# 1.데이터
(x_train, y_train),(x_test, y_test)  = cifar100.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


img_path = "D:/study_data/_data/dog/sheep_dog.PNG"
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img) 
# print(x, "\n", x.shape)  #  (224, 224, 3)


x = preprocess_input(x)      # 데이터를 전처리 해준다. 즉, 스케일링! 

x = np.expand_dims(x, axis=0)   # 차원을 늘려준다 / reshape해도 괜찮음  맨 앞에 행부분을 처리하려고 이걸함  (0, 1, 2 ,3)으로 위치조절 가능
# print(x, "\n", x.shape)   #  (1, 224, 224, 3)
# print(np.min(x), np.max(x))   # 최소값과 최대값 출력
# -98.779 75.061


#2. 모델구성
keras_model = VGG19(weights="imagenet", include_top=False,  
                input_shape=(32, 32, 3))

model = Sequential()
model.add(keras_model) 
model.add(Flatten())
model.add(Dense(128, activation='relu'))  
model.add(Dense(64, activation='relu'))  
model.add(Dense(100, activation="softmax")) 


#3. 컴파일, 훈련
optimizer = "adam"

model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')


import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau # ReduceLROnPlateau : learning_rate를 감축해줌
es = EarlyStopping(monitor="val_loss", patience=20, mode="min", verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=10, model="auto", verbose=1, factor=0.5) # factor=0.5 : 50%만큼 learning_rate를 감축해줄거야
                                                                                                    # 0.001 (50%감축)-> 0.0005 (50%감축)-> 0.00025  즉, 조건이 거릴ㄹ 때 마다 50%씩 감축해준다.
                                                                                                    # 이렇게 흘러가다가 얼리스타핑 걸리면 바로 훈련 정지시키는 원리임!
start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.4,
          batch_size=128, callbacks=[es, reduce_lr])
end = time.time()


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print("loss :",  loss)
print('acc : ', acc)

print('걸린시간 : ' , end - start)


# loss : 6.756853103637695
# acc :  0.2451999932527542
# 걸린시간 :  539.6910088062286








