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

from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score


# 1. 데이터
datasets = load_diabetes()

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

# acc:  [0.15450648 0.11477649 0.16549255 0.04948252 0.05892642] 
#  cross_val_score:  0.1086
# [134.35890141 131.59442394 138.41287365 137.67308617 125.47440062
#  123.14722842 141.22934741 122.73849348 139.36816112 136.2156986
#  135.42946328 116.62837524 135.13479328 120.3791922  121.72776332
#  121.71682395 141.02625793 135.36566926 133.74467964 121.32399604
#  141.12905464 134.5317254  119.33364352 133.33201058 134.50151238
#  142.21146209 120.28931932 136.15120826 121.63320967 119.30048267
#  131.35621187 121.64199351 143.41581479 121.54950897 122.55751394
#  130.29836843 138.42528403 135.73254344 129.79924189 140.6919682
#  128.3614614  116.32241817 119.35553315 122.04556211 142.78128074
#  132.9759     130.22566644 133.5631006  138.19304149 127.77245726
#  133.95030957 125.03790912 127.02404907 121.77111758 125.39626024
#  128.49669174 133.16161688 140.05016322 126.3450541  136.43668578
#  128.24238766 120.29125313 141.00025735 140.92711569 137.91408642
#  131.19675189 129.15570969 121.48278688 129.64830598 134.26750466
#  119.06432584 119.33248563 141.81931116 127.40558507 138.36158828
#  118.98542841 113.22371893 120.90048462 138.74708727 121.22782125
#  134.81197213 142.93717743 131.61560926 130.62276887 140.50795427
#  131.83591241 139.6730586  130.85288116 135.55335545]
# cross_val_predict r2:  -0.08382119491779783