# 맹그러봐
# optimizer, learning_rate 갱신!!!!!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import time

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)


x_train = x_train.reshape(60000, 28, 28, 1)  # (60000, 28, 28) (60000,)
x_test = x_test.reshape(10000, 28, 28, 1)   # (10000, 28, 28) (10000,)



from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델링 
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4, 4),    
                 padding='same', 
                 input_shape=(28, 28, 1)))      
model.add(MaxPooling2D(2, 2))           
model.add(Dropout(0.2))
model.add(Conv2D(64, (4, 4), padding='valid', activation='relu'))                
model.add(MaxPooling2D(2, 2))          
model.add(Dropout(0.2))
model.add(Conv2D(64, (4, 4), padding='same', activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())    # (N, 63)  (N, 175)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.summary()



#3. 컴파일, 훈련
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop, nadam


# learning_rate = 0.0001
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.5, 0.05, 0.005, 0.0005]
for lr in learning_rates:
      optimizer1 = adam.Adam(learning_rate=lr)
      optimizer2 = adadelta.Adadelta(learning_rate=lr)
      optimizer3 = adagrad.Adagrad(learning_rate=lr)
      optimizer4 = adamax.Adamax(learning_rate=lr)
      optimizer5 = rmsprop.RMSProp(learning_rate=lr)
      optimizer6 = nadam.Nadam(learning_rate=lr)

      optimizers = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5, optimizer6]

      for optimizer in optimizers:
            # model.compile(loss='mse', optimizer='adam')
            model.compile(loss='mse', optimizer=optimizer)  # optimizer learning_rate=0.001 디폴트

            model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=0)

            #4. 평가, 예측
            loss = model.evaluate(x_test, y_test)
            y_predict = model.predict([11])

            optimizer_name = optimizer.__class__.__name__
            print('loss : ', round(loss, 4), 'lr : ', lr, 
                  '{0} 결과물 : {1}'.format(optimizer_name, y_predict))

