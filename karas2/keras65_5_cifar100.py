# trainable = True, False 비교해가면서 만들어서 결과 비교

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import VGG16
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from keras.applications import VGG19

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

keras_model = VGG19(weights="imagenet", include_top=False,  
                input_shape=(32, 32, 3))


model = Sequential()
model.add(keras_model) 
model.add(Flatten())
model.add(Dense(128, activation='relu'))  
model.add(Dense(64, activation='relu'))  
model.add(Dense(100, activation="softmax")) 

model.trainable = True

model.summary()          # CNN연산 : 3 x 3 (필터 크기) x 32 (#입력 채널) x 64(#출력 채널) + 64 = 18496 입니다.  / https://gaussian37.github.io/dl-keras-number-of-cnn-param/
                                    # Trainable:True  / VGG False /  model False
print(len(model.weights))            # 30 / 30 / 30    <-- len이므로 weights의 갯수임
print(len(model.trainable_weights))  # 30 /  4 /  0


######################### 2번 소스에서 아래만 추가 ########################################

# print(model.layers)

import pandas as pd
pd.set_option("max_colwidth", -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]

results = pd.DataFrame(layers, columns=["Layer Type", "layer Name", "Layer Trainable"])
print(results)


#3. 컴파일, 훈련
model.compile(optimizer='adam', metrics=['accuracy'],
                loss='sparse_categorical_crossentropy')
from tensorflow.python.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor="val_loss", patience=35, mode="min", verbose=1)
model.fit(x_train, y_train, epochs=300, validation_split=0.4, verbose=1,
          batch_size=128, callbacks=[es])

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)


"""
trainable = True, False 비교해가면서 만들어서 결과 비교
 - model.trainable의 디폴트는 = True   /  False로  설정하면 모델의 trainable을 시키지않겠다 라는 뜻이다.
"""
### model.trainable = False ##########
# 235/235 [==============================] - 9s 29ms/step - loss: 16.7880 - accuracy: 0.1105 - val_loss: 16.6798 - val_accuracy: 0.1151
# Epoch 2/300
# 235/235 [==============================] - 6s 26ms/step - loss: 16.7880 - accuracy: 0.1105 - val_loss: 16.6798 - val_accuracy: 0.1151
# Epoch 3/300
# 235/235 [==============================] - 6s 26ms/step - loss: 16.7880 - accuracy: 0.1105 - val_loss: 16.6798 - val_accuracy: 0.1151

# 가중치를 동결시켜서 가중치가 갱신되지 않음

# 2022-09-10 13:25:45.983045: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8204
# 235/235 [==============================] - 9s 29ms/step - loss: 22.5889 - accuracy: 0.0121 - val_loss: 22.6743 - val_accuracy: 0.0118
# Epoch 2/300
# 235/235 [==============================] - 6s 26ms/step - loss: 22.5889 - accuracy: 0.0121 - val_loss: 22.6743 - val_accuracy: 0.0118
# Epoch 3/300
# 235/235 [==============================] - 6s 26ms/step - loss: 22.5889 - accuracy: 0.0121 - val_loss: 22.6743 - val_accuracy: 0.0118



### model.trainable = True ##########
# loss :  4.59420919418335
# acc :  0.31150001287460327
