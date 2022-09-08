"""
[해당 프로젝트 설명]
https://colab.research.google.com/drive/1EigZPRIZhmGWcWI5r--hoAaAFkqb92aG



Epoch 1/5
60/60 [==============================] - 1s 4ms/step - loss: 0.9166 - acc: 0.7268
# 60/60의 뜻 : 전체데이터 % 배치사이즈 - 발리데이션_스플릿 = 45가 0.6 + 발리데이션 0.4 = 60
# 전체데이터 % 32 = 60  -->  60 - 15 = 45


"""


import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input,Dropout
import keras
import tensorflow as tf 

# 1.데이터
(x_train,y_train),(x_test,y_test)  = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.


# from keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# 2.모델 
def build_model(drop=0.5, optimizer='adam', activation='relu', node1=111):
    inputs = Input(shape=(28*28), name="input")
    x = Dense(512, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name="hidden2")(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=activation, name="hidden3")(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation="softmax", name="outputs")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')
    return model

def create_hyperparameter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu','linear','sigmoid','selu','elu']
    return {'batch_size': batchs, 'optimizer' : optimizers, 'drop' : dropout, 'activation': activation}

hyperparameters = create_hyperparameter()
# print(hyperparameters)

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
keras_model = KerasClassifier(build_fn=build_model, verbose=1)
#케라스(텐서플로)모델을 사이킷런 모델로 래핑해준다.

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
model = RandomizedSearchCV(keras_model, hyperparameters, cv=2, n_iter=3, verbose=1)  # 디폴트 : 10
  # cv는 2이상  /  n_iter도 2이상  n_iter는 cv이거를 검증하는 거 같음 찾아볼 것


import time
start = time.time()
model.fit(x_train,y_train, epochs=5, validation_split=0.4)   
end = time.time()

print('걸린시간 : ' , end - start)
print("model.best_param_: " , model.best_params_)    
print("model.best_estimator_: " , model.best_estimator_)    
print("model.best_score_: " , model.best_score_)    
print('model.score : ', model.score)


from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc : ', accuracy_score(y_test,y_predict))


# 걸린시간 :  42.399972438812256
# model.best_param_:  {'optimizer': 'adam', 'drop': 0.3, 'batch_size': 300, 'activation': 'selu'}
# model.best_estimator_:  <keras.wrappers.scikit_learn.KerasClassifier object at 0x0000017655DFE2B0>
# model.best_score_:  0.9371999800205231
# model.score :  <bound method BaseSearchCV.score of RandomizedSearchCV(cv=2,
#                    estimator=<keras.wrappers.scikit_learn.KerasClassifier object at 0x0000017654EEFDF0>,
#                    n_iter=3,
#                    param_distributions={'activation': ['relu', 'linear',
#                                                        'sigmoid', 'selu',
#                                                        'elu'],
#                                         'batch_size': [100, 200, 300, 400, 500],
#                                         'drop': [0.3, 0.4, 0.5],
#                                         'optimizer': ['adam', 'rmsprop',
#                                                       'adadelta']},
#                    verbose=1)>
# 313/313 [==============================] - 1s 2ms/step
# acc :  0.9573