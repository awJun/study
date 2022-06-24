# 데이콘 따릉이 문제풀이
import numpy as np
import pandas as pd
  # pandas = 데이터를 불러올 때 사용 (이외에도 기능도 많다.)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
  # mean_squared_error: mse이다. 여기서 루트를 하면 rmse가 된다.

#1. 데이터
# "". 현재폴더 / 하단에" 라는 뜻
path = './_data/ddarung/'   # <- 폴더 경로만 있다. 폴더 안에 어던 데이터셋을 가져올지 선언은 안함
                            # 그러므로 'train.csv' 를 통해서 데이터를 가져옴
                            
                            
train_set = pd.read_csv(path + 'train.csv', index_col=0) 
print(train_set)
print(train_set.shape) # [1459 rows x 10 columns]
                       # (1459, 10)

# test_set는 예측에서 쓸 예정 ~
test_set = pd.read_csv(path + 'test.csv', index_col=0) 
print(test_set)
print(test_set.shape)  # [715 rows x 9 columns]
                       # (715, 9)

"""
    RMSE는 제곱하면 값이 너무 커지므로 루트를 씌워서 값을 다시 줄이는 것

대회사이트 링크  https://dacon.io/competitions/open/235576/data




train.csv 안에서 데이터셋과 테스트셋을 모두 알아서하여 회귀모델을 만들고 test.csv로 y 제출

데이터셋 안에 맨뒤에 있는 count은 결과값이므로 제외시키고 사용
맨 앞은 index_col으로 앞으로 뺌

"""

print(test_set.columns)  # 컬럼의 이름을 알려줌
print(train_set.info())  # non-null -> 데이터가 빠져있다.   null 값이없다          
                         # 이빨이 빠진 데이터 이것을 결측치 라고한다.

print(train_set.describe())  # describe: 서술하다 묘사하다.  해당 값 묘사해서 알려줌

#### 결측치 처리 1. 제거 ####
print(train_set.isnull().sum()) # null의 합을 구하겠다.
train_set = train_set.dropna()  # nall이 있는 행 부분을 전체 삭제 
                                # 해당 행 부분의 데이터를 다 삭제하므로 데이터 손실이 크다.
print(train_set.isnull().sum())
print(train_set.shape)          # (1328, 10)   <- 살아남은 데이터양
############################


  # x 선언
x = train_set.drop(['count'], axis=1)  # drop: 빼버린다. count 제거

print(x)
print(x.columns)
print(x.shape)  # (1459, 9) inpip dim = 9  <- 원래 데이터양은 위에 train_set.isnull().sum()를 
                #                             거쳐서 데이터가 줄었다. (현재는 이 값이 아니다.)
                                               
                                               

  # y 선언
y = train_set['count']  # x에 대한 해답지인 count

print(y)
print(y.shape)   # (1459,) output = 1개





x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=91
                                                    )

# 82(56),  85(54), 91(54) 



#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=9))  # 첫번째 히든에 1넣으면 성능 아작난다.
model.add(Dense(30))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=30)

#4. 결과, 예측
loss = model.evaluate(x_train, y_train)
print('lose : ', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
  # return np.sqrt(mean_squared_error(y_test, y_predict))를 반환한다.
  # sqrt는 루트이다 (mean_squared_error(y_test, y_predict))에 루트(제곱, 제곱근)를 함
  # 100에 루트를하면 10    /  루트를하면 제곱이 줄어든다.
  
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

"""

lose :  2954.836669921875
RMSE :  49.47759449079721

"""




"""
y_predict = model.predict(test_set)

# lose :  2979.27587890625

과제1. 함수에 대해서 공부  (함수는 재사용하기 위해서 만드는 것)
과제2. 결과 제출
"""



























