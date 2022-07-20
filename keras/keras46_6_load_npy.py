import numpy as np
from keras.preprocessing.image import ImageDataGenerator


#1. 데이터

# np.save('d:/study_data/_save/keras46_05_train_x.npy', arr=xy_train[0][0])
# np.save('d:/study_data/_save/keras46_05_train_y.npy', arr=xy_train[0][1])
# np.save('d:/study_data/_save/keras46_05_test_x.npy', arr=xy_test[0][0])
# np.save('d:/study_data/_save/keras46_05_test_y.npy', arr=xy_test[0][1])

x_train = np.load('d:/study_data/_save/_npy/keras46_05_train_x.npy')
y_train  = np.load('d:/study_data/_save/_npy/keras46_05_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras46_05_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras46_05_test_y.npy')

print(x_train)
print(x_train.shape)


# #2. 모델구성
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

# model = Sequential()
# model.add(Conv2D(32, (2,2), input_shape=(100, 100, 1), activation='relu'))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))


# #3. 컴파일, 훈련
# model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])   # 
# # model.fit(xy_train[0][0], xy_train[0][1])  # 현재 x, y데이터 한덩어리씩 있으므로 이렇게함   해당은 배치사이즈를 최대로 할 경우 가능하다.

# # 위에 문제를 해결하기위해 fit_generator를 사용
# hist = model.fit_generator(xy_train, epochs=30,  #  그냥 데이터 통채로 넣으면된다
#                    validation_data=xy_test, 
#                    validation_steps=4,
#                    #[중요!] validation_data=xy_test, validation_steps=4 를 사용해서 아래에서 accuracy = hist.history['accuracy']를 사용하기
#                           # 위해서 선언함 / 선언해야지 metrics=['accuracy']와 만나서 아래에서 결과를 도출해줌 !
                  
#                    steps_per_epoch=33,   # batch_size 대신 이걸 사용해야함   /  1 epoch당 훈련하는 양을 조절하는 것임!
#                     # 범위는 데이터셋을 batch사이즈로 나눈거 즉!  -->  전체데이터/batch = 160/5 = 32 [정정]
#                       # 지금보니까 그냥 제한 범위없음 마음대로 할 것 근데 통상적으로 위에처럼 사용한다고함.
#                     # [중요] fit_generator에서는 batch_size를 사용할 수 없다 ~
#                     )  

# accuracy = hist.history['accuracy']
# val_accuracy = hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# print('loss : ', loss[-1])   # .history를 사용해서 로스값이 여러개가 출력이 되지만 그 중에서 제일 마지막것을 가져와서 보기위해 [-1를]사용 출력된 로스값 데이터들은
#                                   # 리스트 형태로 담기므로 인덱스 번호가 먹는다 !
# print('val_accuracy : ', val_accuracy[-1])
# print('accuracy : ', accuracy[-1])
# print('val_loss : ', val_loss[-1])

# # loss :  0.6927464604377747
# # val_accuracy :  0.550000011920929
# # accuracy :  0.5515151619911194
# # val_loss :  0.6897107362747192


# # 그림그려!!! 

# import matplotlib.pyplot as plt

# plt.figure(figsize = (9,6))     # figsize(가로길이,세로길이)
# plt.plot(hist.history['accuracy'], marker='.', color='green', label='accuracy')           # label='loss' 해당 선 이름
# plt.plot(hist.history['val_accuracy'], marker='.', color='yellow', label='val_accuracy')  # marker='.' 점으로 찍겟다
# plt.plot(hist.history['loss'], marker='.', color='red', label='loss')  # marker='.' 점으로 찍겟다
# plt.plot(hist.history['val_loss'], marker='.', color='blue', label='val_loss')  # marker='.' 점으로 찍겟다
# plt.grid()                        # plt.grid(True)    # grid: 그리다
# plt.title('ha_ha_ha')                # title의 이름을 asaql로 하겠다
# plt.ylabel('loss')                # y라벨의 이름을 loss로 하겠다
# plt.xlabel('epochs')              # x라벨의 이름을 epochs로 하겠다
# plt.legend(loc = 'upper right')   # 그래프가 없는쪽에 알아서 해준다 굳이 명시할 필요 없음
# # plt.legend()   # 그래프가 없는쪽에 알아서 해준다 굳이 명시를 안 할 경우 사용법
# plt.show()    # 그래프를 보여줘라

