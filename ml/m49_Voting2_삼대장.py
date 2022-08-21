"""
[핵심]

#2. 모델
###[ 이 셋이 삼대장이여 ~ ]###########
xg = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier(verbose=0)
#######################################

위에서 삼대장을 불러와서 

model = VotingClassifier(estimators=[('xg', xg), ('lg', lg), ('cat', cat)],
                         voting='soft',   # hard도 있다.
                         )

로 해주고 m49_Voting1과 동일하게 하면된다.

하지만 현재 출력을 할때 cat에서 나오는 verbose 때문에 출력한 결과를 보기 어려운 상황이므로 
verbose를 안뜨게하거나 cat의 순서를 제일 앞으로 이동시켜서 verbose를 결과값 위에 뜨게하면 해결된다.

### 1에서 사용한 이부분을 사용하면 Cat에서 나오는 verbose 때문에 나머지가 안보이는 문제점이있음######
# 그래서 cat의 순서를 [xg, lg, cat]에서 [cat, xg, lg] 순서를 변경하거나 verbose를 안뜨게하면 된다.

# verbose 안뜨게 하는 방법은 CatBoostClassifier(verbose=0)부분에서 verbose=0을 해주면 된다.







from sklearn.ensemble import VotingClassifier

lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)

model = VotingClassifier(estimators=[('LR', lr), ('KNN', knn)],
                         voting='soft'   # hard도 있다.
                         )

위에서 VotingClassifier을 설정하고 아래에서 for문으로 위에서 만든 모델 두개를 동시에 돌렸다.

classifiers = [lr, knn]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_predict)
    class_name = model2.__class__.__name__   # .__class__.__name__  모델의 이름을 출력해준다.
    print("{0} 정확도: {1:.4f}".format(class_name, score2))  
    
    # {0} : format 첫번째 위치에 있는 class_name를 출력해라
    # {1:.4f} : format 두번째 위치에 있는 score2를 소수 4번째 자리까지 출력해라



[TMI]
# VotingClassifier soft는 평균 / hard는 0 0 1이면 0 투표 방식임  1 1 0이면 1 
#  통상적으로 soft가 성능이 더 좋다.


print("보킹 결과 : ", round(score, 4))  # 소수 4번째 자리까지 출력이라는 뜻 / 결과 값을 소수 4번째 자리까지 반올림하겠다.

"""


import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier      # pip install lightgbm
from catboost import CatBoostClassifier  # pip install catboost



#1. 데이터 
datasets = load_breast_cancer()

# df = pd.DataFrame(datasets.data, columns=datasets.feature_names)   # x 데이터만 들어감
# print(df.head(7))

x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=123,
                                                    stratify=datasets.target
                                                    )

scaler = StandardScaler()
x_trian = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


#2. 모델
###[ 이 셋이 삼대장이여 ~ ]###########
xg = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier(verbose=0)
#######################################

model = VotingClassifier(estimators=[('xg', xg), ('lg', lg), ('cat', cat)],
                         voting='soft',   # hard도 있다.
                         )

# VotingClassifier soft는 평균 / hard는 0 0 1이면 0 투표 방식임  1 1 0이면 1 
#  통상적으로 soft가 성능이 더 좋다.

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

score = accuracy_score(y_test, y_predict)

print("보킹 결과 : ", round(score, 4))  # 소수 4번째 자리까지 출력이라는 뜻 / 결과 값을 소수 4번째 자리까지 반올림하겠다.
# 보킹 결과 :  0.9737


classifiers = [cat, xg, lg]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_predict)
    class_name = model2.__class__.__name__   # .__class__.__name__  모델의 이름을 출력해준다.
    print("{0} 정확도: {1:.4f}".format(class_name, score2))  
    
    # {0} : format 첫번째 위치에 있는 class_name를 출력해라
    # {1:.4f} : format 두번째 위치에 있는 score2를 소수 4번째 자리까지 출력해라





### 1에서 사용한 이부분을 사용하면 Cat에서 나오는 verbose 때문에 나머지가 안보이는 문제점이있음######
# 그래서 cat의 순서를 [xg, lg, cat]에서 [cat, xg, lg] 순서를 변경하거나 verbose를 안뜨게하면 된다.

# verbose 안뜨게 하는 방법은 CatBoostClassifier(verbose=0)부분에서 verbose=0을 해주면 된다.

# CatBoostClassifier 정확도: 0.6316











