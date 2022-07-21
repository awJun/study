# 증폭할 때는 평가(test)에서 사용할 데이터는 증폭하면 안됨

from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(   # ImageDataGenerator 이미지를 숫자화(수치화)
    rescale=1./255,   # 나는 1개의 이미지를 픽셀 255개로 나눌거야 스케일링한 데이터로 가져오겠다 ..? 
    horizontal_flip=True,
    # vertical_flip=True,  # 반전시키겠냐 ? / true 네! 라는 뜻이라고함
    width_shift_range=0.1, # 가로 세로
    height_shift_range=0.1, # 상 하
    rotation_range=5, # 돌리겟다?
    zoom_range= 0.1, # 확대
    # shear_range= 0.7, # 선생님 : 알아서 찾아 ~ ;;;  /  선생님 : 찌글찌글 ?? ;
    fill_mode='nearest'  
)

augument_size = 40000   # 4만장을 늘리겠다 ~ 라는 거임   /  이 데이터는 아래에서 60000의 데이터중 랜덤하게 정수를 뽑을 예정임 ~ 
randidx = np.random.randint(x_train.shape[0], size=augument_size)    # https://www.sharpsightlabs.com/blog/np-random-randint/  [np.random.randint 설명링크]
                             # 60000 - 40000 /  60000개에서 40000만개의 데이터를  np.random.randint을 사용해서 정수를 랜덤으로 뽑아서 randidx안에 넣겠다는 뜻.
 # np.random.randint : 랜덤하게 정수를 넣는다.
print(x_train.shape[0])  # 60000, 28, 28, 1 중에서 60000을 출력해주고 [1]일 경우는 28을 출력해준다.
print(randidx.shape)  # (40000,)
print(np.min(randidx), np.max(randidx))   #  최소값--> 2 59996  <-- 최대값
print(type(randidx))   # <class 'numpy.ndarray'>   numpy형태는 기본적으로 리스트 형태이다.

x_augumented = x_train[randidx].copy()   # 연산할 때 새로운 공간을 만들어서 할때는 .copy()를 사용하면 새로운 메모리을 확보해서 그 공간에서 작업을 하겠다는 뜻이다. 
                                              # 즉 ! 원본을 전혀~~ 안건들고 새로운 공간에서 연산을 하겠다는 뜻! / 이것으로 인해서 안전성이 올라갔다
# x_train에서 randidx를 뽑아서 x_augumented에다가 저장하겟다는 뜻
y_augumented = y_train[randidx].copy()   
# y_train에서 randidx를 뽑아서 y_augumented에다가 저장하겟다는 뜻
 
print(x_augumented.shape)    # (40000, 28, 28)
print(y_augumented.shape)    # (40000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_augumented = x_augumented.reshape(x_augumented.shape[0],
                                    x_augumented.shape[1],
                                    x_augumented.shape[2], 1
                                    )
 
x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,         # (38000, 28, 28, 1)
                                  shuffle=False).next()[0] 

print(x_augumented.shape)   # (40000, 28, 28, 1)
# print(x_augumented.shape)    # (40000, 28, 28, 1)
                            
x_train = np.concatenate((x_train, x_augumented))  #엮다          [[과제]클래스 공부해라 ~~  괄호 두 개의 의미를 찾아라]
y_train = np.concatenate((y_train, y_augumented))  
                    
           
                    
print(x_train.shape, y_train.shape)      # (100000, 28, 28, 1) (100000,)             
 

# #==[위에는 2번 긁어옴]=====================================================================================
# # 성능비교, 징폭 전 후 비교

#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape = (28,28,1),activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3.컴파일,훈련
model.compile(loss= 'binary_crossentropy',optimizer='adam')
# batch를 최대로 잡으면 이렇게도 가능.
from tensorflow.python.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor= 'val_loss', patience=30, mode='auto', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, 
                 validation_split=0.2, 
                 verbose=1, 
                 batch_size=32,
                 callbacks=es)
# hist = model.fit_generator(x_train,y_train, epochs= 40, steps_per_epoch=32,
#                                         #전체데이터/batch = 160/5 = 32
#                     validation_data=x_train,
#                     validation_steps=4) #val_steps: 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다. 

# accuracy = hist.history['accuracy']
# val_accuracy =  hist.history['val_accuracy']
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])
# print('accuracy : ', accuracy[-1])
# print('val_accuracy : ', val_accuracy[-1])

#4.평가,예측 
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)   

from sklearn.metrics import accuracy_score
y_predict = y_predict.round()
acc = accuracy_score(y_test,y_predict)

print('loss : ', loss)
print('acc : ', acc)












