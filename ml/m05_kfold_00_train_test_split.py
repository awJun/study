"""
[핵심]
print("y의 라벨값 : ", np.unique(y))  # 해당 데이터의 고유값을 출력해준다.

KFold는 회귀와 분류 둘 다 사용할 수 있다.

cross_val_score()을 사용하기 위에서 괄호 안의 옵션인 KFold를 

n_splits = 5  
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

이런 형태로 작성을 한 후에

scores = cross_val_score(model, x, y, cv=kfold)
print("ACC : ", scores, '\n cross_val_score : ', round(np.mean(scores), 4)) 
                                                # round를 사용해서 소수 4번째까지 출력해라 라고함

이러한 형태로 사용을 한다 
이때 round(np.mean(scores), 4))의 의미는 cross_val_score의 값인  score의 값을 평균값으로 변경한 후
np를 이용해서 반올림 과정을 거친 후 소수 4번째 자리까지 출력하라는 의미이다.

[ cross_val_score 사용하는 이유 ]
기존의 validation은 train 데이터에서 손해를 보면서 분할을하여 모델 검증 데이터로 사용을 하였는데
cross_val_score를 사용하면 전체 데이터에서 n_splits 안에 들어있는 변수의 숫자만큼 나누고 fit을 할 때
마다 나눈 부분을 순차적으로 validation 데이터로 사용한다. 이로인해 train 데이터의 손실을 하지않고
validation의 역할을 수행할 수 있다. 만약 n_splits = 5 라면 전체 데이터를 5개로 분할하여 순차적으로
validation 데이터로 사용하게된다.

cross_val_score를 사용하면 train_test_split을 사용할 필요가 없다.
cross_val_score(model, x, y, cv=kfold) 형태로 사용되기 때문이다.
cross_val_score(model, x, y, cv=kfold) 과정에서 fit과 컴파일도 같이 진행된다.


"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=99)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)
                      
#2. 모델구성
from sklearn.svm import SVC
model = SVC()
    
# 3. 4. 컴파일, 훈련, 평가, 예측
# model.fit(x_train, y_train)
score = cross_val_score(model, x_train, y_train, cv=kfold)
# score = cross_val_score(model, x, y, cv=5)
# 이렇게 하면 위에서 kFold로 따로 정의하지 않아도 된다 대신 파라미터들을 건들 수 있는게 줄어듦
print('acc: ', score, '\n cross_val_score: ', round(np.mean(score),4))

y_predict = cross_val_score(model, x_test, y_test, cv=kfold)
print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict acc: ', acc)

# acc:  [0.95652174 1.         1.         1.         0.95454545]
#  cross_val_score:  0.9822
# [0.875 1.    0.875 1.    1.   ]