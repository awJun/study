"""
[핵심]
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

from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict


# 1. 데이터
datasets = load_boston()

x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)

#2. 모델구성
from sklearn.svm import SVR
model = SVR()

# 3. 4. 컴파일, 훈련, 평가, 예측
# model.fit(x_train, y_train)
score = cross_val_score(model, x_train, y_train, cv=kfold)
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test, y_predict)

print('acc: ', score, '\n cross_val_score: ', round(np.mean(score),4))
print(y_predict)
print('cross_val_predict r2: ', r2)

# acc:  [0.16266513 0.11966027 0.02766872 0.28537389 0.24316849] 
#  cross_val_score:  0.1677
# [17.38663628 15.14895542 23.19927814 16.93580155 23.23487275 18.2672927
#  23.23233053 23.04018678 17.28548534 18.33825646 22.75416708 16.13797786
#  21.76553387 15.94285808 20.08944035 22.81097964 17.21260641 15.67902476
#  23.51659191 15.59630947 22.21315302 15.69493626 22.20583756 17.98967878
#  22.32950495 23.50650525 22.40963276 14.04962681 20.98591908 22.14549412
#  14.77605034 14.61172464 22.97876916 21.37556349 20.43335368 20.28169043
#  22.16443616 22.60784745 17.30346832 23.1257789  20.94453539 22.24008445
#  14.87769231 23.31690815 23.25295559 23.27604331 22.79207928 22.43141099
#  17.05623346 23.39138493 23.03686638 22.57219288 15.09639892 22.43767486
#  20.55367541 15.20051435 22.20291092 15.08050931 23.32517963 22.08462991
#  22.40458724 22.39812528 15.41758061 14.53136568 23.21785405 22.24170953
#  21.88317675 22.28890912 15.08500136 22.26402227 22.61011495 17.26554118
#  22.29329924 22.09358725 17.29030685 22.39184291 20.06660113 23.30182581
#  20.90203094 15.35971261 17.32039099 22.25415627 22.64590894 19.62782479
#  20.92697568 22.37020125 22.35607405 14.57370124 22.0732672  17.25824094
#  20.53965133 16.60673014 21.33412006 22.42916756 21.95041733 23.06989848
#  22.30100534 20.4691103  23.37563452 20.35463326 22.7447506  22.66290704]
# cross_val_predict r2:  0.15621531230049834