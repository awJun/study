import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston   # sklearn은 학습 예제가 많음

#1. 데이터
datasets = load_boston()   
x = datasets.data    # 데이터가
y = datasets.target  # y에 들어간다.

'''
 - 전처리가 된 데이터
[[6.3200e-03 1.8000e+01 2.3100e+00 ... 1.5300e+01 3.9690e+02 4.9800e+00]
 [2.7310e-02 0.0000e+00 7.0700e+00 ... 1.7800e+01 3.9690e+02 9.1400e+00]
 [2.7290e-02 0.0000e+00 7.0700e+00 ... 1.7800e+01 3.9283e+02 4.0300e+00]
 ...
 [6.0760e-02 0.0000e+00 1.1930e+01 ... 2.1000e+01 3.9690e+02 5.6400e+00]
 [1.0959e-01 0.0000e+00 1.1930e+01 ... 2.1000e+01 3.9345e+02 6.4800e+00]
 [4.7410e-02 0.0000e+00 1.1930e+01 ... 2.1000e+01 3.9690e+02 7.8800e+00]]
'''

'''
1과 100을 0과 1로 압축한다 (동일한 비율로) 
0은 1    100은 1로 압축

데이터가 적으면 상관없지만 대부분 데이터가 대용량이므로 압축을하여 효율을 극대화 시킨다.

데이터처리 친구 덴첼




'''

print(x)
print(y)

print(x.shape, y.shape) # (506, 13) (506,)  열 13    (506, ) 506개 스칼라, 1개의 백터
                        # intput (506, 13), output 1
print(datasets.feature_names)
 # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 # 'B' 'LSTAT']
 
print(datasets.DESCR)
 
 
# [실습] 아래를 완성할 것
# 1. train 0.7
# 2. R2 0.8 dltkd


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=12345678
                                                    )



#2. 모델구성
model = Sequential()
model.add(Dense(300, input_dim=13))
model.add(Dense(240))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(220))
model.add(Dense(220))
model.add(Dense(1))

#3. 컴파일. 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=3000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   
print('r2스코어 : ', r2)


loss :  3.3255813121795654
r2스코어 :  0.711256904523248







































































