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

from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold, cross_val_score


# 1. 데이터
datasets = load_diabetes()

x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

n_splits = 2
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)

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

# acc:  [0.07758525 0.07779418]
#  cross_val_score:  0.0777
# [120.68763444 135.85620795 125.73995124 137.51177415 121.75101796
#  140.53140708 124.66511827 135.54458426 123.39778092 120.34957207
#  136.5989599  122.04429158 135.51762101 126.45728829 133.56567657
#  119.88138752 124.53413993 134.52323775 119.7384368  138.7772557
#  123.97252364 135.58838596 124.94610658 136.74135523 120.14615153
#  125.78424014 125.68970846 136.18678161 118.70507018 139.94380532
#  126.73617413 140.18193747 126.27744303 140.8406119  120.52269461
#  134.7714531  123.6409952  135.18000633 124.01514032 138.93932352
#  118.75503123 136.16443837 124.03927843 133.81652947 124.78945038
#  134.18375532 133.16615442 119.36227669 140.07805637 123.87893734
#  134.98797547 121.46621762 132.5338793  134.29971448 121.99276207
#  133.63451689 119.99304434 140.94225625 137.85771485 138.92964528
#  124.34046479 138.64603993 125.65731209 124.38050364 136.63160071
#  120.9311098  139.12376537 119.04566775 134.65263764 137.12892595
#  125.05708    137.15224829 139.33608478 118.23503983 122.95183523
#  123.72566058 133.46719851 126.22559106 136.90820499 140.42028382
#  120.63978868 126.22522627 122.56475268 139.55430023 138.22843371
#  118.32668103 137.81669672 121.82914975 138.45043413]
# cross_val_predict r2:  -0.01720049106619914

# ValueError: n_splits=5 cannot be greater than the number of members in each class.