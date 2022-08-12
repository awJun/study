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

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
                        
#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)  #행(Instances): 150   /   열(Attributes): 4
# print(datasets.feature_names)

x = datasets['data']  # .data와 동일 
y = datasets['target']  
# print(x.shape)   # (150, 4)
# print(y.shape)   # (150,)
# print("y의 라벨값 : ", np.unique(y))  # 해당 데이터의 고유값을 출력해준다.


# x_train, x_test, y_train, y_test = train_test_split(x, y,
#                                                     train_size=0.8,
#                                                     shuffle=True,
#                                                     random_state=100
#                                                     )

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression는 분류임
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = SVC()


#3.4 훈련,  컴파일, 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold)
print("ACC : ", scores, '\n cross_val_score : ', round(np.mean(scores), 4))  # round를 사용해서 소수 4번째까지 출력해라 라고함

# ACC :  [0.96666667 0.96666667 1.         0.93333333 0.96666667] 
#  cross_val_score :  0.9667








