"""
[핵심]
StratifiedKFold는 분류에서만 사용할 수 있다.

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

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict


# 1. 데이터
datasets = load_digits()

x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)

#2. 모델구성
from sklearn.svm import SVC
model = SVC()
    
# 3. 4. 컴파일, 훈련, 평가, 예측
# model.fit(x_train, y_train)
score = cross_val_score(model, x_train, y_train, cv=kfold)
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_predict)

print('acc: ', score, '\n cross_val_score: ', round(np.mean(score),4))
print(y_predict)
print('cross_val_predict acc: ', acc)

# acc:  [0.98611111 0.98958333 0.97212544 0.97909408 1.        ] 
#  cross_val_score:  0.9854
# [1 1 7 2 4 0 1 8 8 3 1 0 5 3 6 2 3 8 2 5 3 5 0 0 6 8 3 2 3 8 0 1 3 2 8 0 1
#  7 1 3 9 2 1 4 1 1 2 8 4 4 0 2 8 4 8 5 7 3 1 5 9 2 4 1 5 2 0 5 1 4 8 4 7 6
#  1 9 5 1 7 6 4 0 2 5 9 1 9 7 8 7 6 4 1 5 3 4 8 8 2 6 7 9 4 1 6 4 0 5 7 8 1
#  3 4 3 1 3 8 6 1 5 0 7 8 9 0 1 9 7 5 6 7 8 9 2 4 3 8 9 0 5 2 2 1 5 1 0 1 8
#  5 5 4 5 2 5 1 7 5 5 7 4 9 3 5 4 6 9 0 3 4 1 6 0 6 3 2 8 3 9 2 2 2 8 7 4 2
#  2 8 3 7 4 2 8 5 0 1 8 9 0 7 5 1 6 9 0 7 5 1 3 7 3 0 9 2 7 9 8 9 4 0 7 8 3
#  5 3 4 6 6 5 0 9 6 0 6 9 4 1 5 5 0 4 2 2 2 3 4 0 8 0 9 4 5 1 4 1 3 8 4 4 2
#  8 2 2 7 1 8 2 0 2 9 6 2 9 3 7 4 5 7 4 7 5 6 4 5 9 2 9 1 6 9 9 5 2 0 5 6 1
#  7 1 3 4 1 5 8 1 2 1 2 6 4 4 9 9 2 3 5 1 2 7 6 3 7 1 6 4 8 0 2 8 4 6 1 7 3
#  0 0 9 1 9 2 1 9 8 2 6 6 0 6 2 7 0 4 6 5 5 7 8 3 3 8 3]
# cross_val_predict acc:  0.9555555555555556