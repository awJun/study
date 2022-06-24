import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston   # sklearn은 학습 예제가 많음



#1. 데이터
datasets = load_boston()   
x = datasets.data    # 데이터가
y = datasets.target  # y에 들어간다.

print(x)
print(y)

print(x.shape, y.shape) # (506, 13) (506,)  열 13    (506, ) 506개 스칼라, 1개의 백터
                        # intput (506, 13), output 1
                        
print(datasets.feature_names)
 # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 # 'B' 'LSTAT']
 
print(datasets.DESCR)


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

import time

#3. 컴파일. 훈련
model.compile(loss='mae', optimizer='adam')

start_time = time.time()  # 현재 시간을 출력해준다
print(start_time)         # 1656032959.3420238

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=0)

end_time = time.time() - start_time
print("걸린시간 : ", end_time)

"""
verbose는 출력 과정을 보여주는 여부를 선택하는 것 (보이진 않지만 프로그램은 계속 돌아가므로 cpu는
                                                계속 움직인다.) 

프로그램이 돌아가는 것이 안보이므로 본인판단

verbose 0
걸린시간 :  24.135599613189697  / 출력없다.

verbose 1
걸린시간 :  29.90717315673828   / 잔소리 많다.
 
verbose 2
걸린시간 :  24.71705937385559   / 프로그래스바 없다.

verbose 3, 4, 5... / 3이후로는 동일하다.
걸린시간 :  24.686833381652832  / epoch만 나온다.

"""


""" 
파라미터 매개변수 (함수 안에서 사용)
verbose => 훈련과정을 안보여주고 결과만 보여준다.
verbose 사용하는 이유 사람이 보려면 강제 지연을 해야하는데 이 지연 과정에서 강제로 느려지는 것을 
방지하기 위함이다.
"""























