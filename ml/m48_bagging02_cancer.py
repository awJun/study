"""
[핵심]
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression  # 2진 분류에서 사용   /  linear_model은 제일 간단한 모델임

model = BaggingClassifier(LogisticRegression(),
                          n_estimators=100,    # 생성할 의사결정 나무 개수
                          n_jobs= -1,
                          random_state=123
                          )

배깅이라는 것은 한가지 모델을 여러번 돌려서 만든 것이 배깅이다.
즉! 그리드서치랑 비슷한? 느낌같음 아마도..? 이 줄은 그냥 무시해줘...

[주의점]
BaggingClassifier은 스케일링을 하지 않으면 에러가 발생한다.

"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

datasets = load_breast_cancer()
x, y, = datasets.data, datasets.target
print(x.shape, y.shape)     # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=123,
                                                    stratify=y
                                                    )


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) 
x_test = scaler.fit_transform(x_test) 


#.2 모델
###[ 이번 핵심! ]####################################
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression  # 2진 분류에서 사용   /  linear_model은 제일 간단한 모델임
from xgboost import XGBClassifier, XGBRegressor

model = BaggingClassifier(XGBClassifier(n_estimators=100,
                          learning_rate=1,
                          max_depth=2,
                          gamma=0,
                          min_child_weight=1,
                          subsample=1,
                          colsample_bytree=0.5,
                          colsample_bylevel=1,
                          colsample_bynode=1,
                          reg_alpha=0.01,
                          reg_lambd=1,
                          tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,
                          ),
                          verbose=1,
                          n_estimators=100,    # 생성할 의사결정 나무 개수
                          n_jobs= -1,
                          random_state=123,
                          )

# cv 사용은 다 가능하다. !! 내가 못사용했던것임.

######################################################

#.3 훈련
model.fit(x_train, y_train)
# BaggingClassifier은 스케일링을 하지 않으면 에러가 발생한다.

#.4 평가, 예측
print(model.score(x_test, y_test))



# LogisticRegression 결과
# 0.9824561403508771

# XGBClassifier 결과
# 0.9883040935672515

















