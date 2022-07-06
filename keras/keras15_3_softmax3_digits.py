import numpy as np
import time

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import load_digits

#.1 데이터
datasets = load_digits()
x = datasets.data     # (1797, 64)
y = datasets.target   # (1797,)

'''
print(x.shape)        # (1797, 64)
print(y.shape)        # (1797,)
print(np.unique(y, return_counts=True)) # (10) 고유
print(np.unique(x, return_counts=True)) # (13) 고유
'''
# (1797, 64)
# (1797,)
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
# array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))


# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[10])
# plt.show()


from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
print(y)
print("+++++++++++++++++++++++++")
print(x)

print(x.shape) # (1797, 64)
print(y.shape)   # (1797, 10)



# x: 17  /  y: 2
# print("y의 라벨값 : ", np.unique(x))


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )



#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=64))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax')) 

#3. 컴파일. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy']) 

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,
                              restore_best_weights=True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=100,
                 verbose=1,
                 validation_split=0.2,
                 callbacks=[earlyStopping])  
end_time = time.time()



#4. 평가, 예측

################################################################################
loss, acc = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력
                                            # loss = loss / acc = metrics에서 나온 accuracy 값
print('loss : ', loss)
print('acc : ', acc)



result = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
print(y_predict)

y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)


# loss :  0.16383910179138184
# acc :  0.9611111283302307
# accuracy :  0.9611111111111111

# model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 100)               6500
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               10100
# _________________________________________________________________
# dense_2 (Dense)              (None, 100)               10100
# _________________________________________________________________
# dense_3 (Dense)              (None, 100)               10100
# _________________________________________________________________
# dense_4 (Dense)              (None, 10)                1010
# =================================================================
# Total params: 37,810
# Trainable params: 37,810
# Non-trainable params: 0
# _________________________________________________________________









