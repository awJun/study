import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=123,
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) 
x_test = scaler.fit_transform(x_test) 

#.2 모델
###[ 이번 핵심! ]####################################
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression  # 2진 분류에서 사용   /  linear_model은 제일 간단한 모델임
from xgboost import XGBClassifier, XGBRegressor

model = BaggingRegressor(LinearRegression(),
                          verbose=1,
                          n_estimators=100,    # 생성할 의사결정 나무 개수
                          n_jobs= -1,
                          random_state=123,
                          )

######################################################

#.3 훈련
model.fit(x_train, y_train)
# BaggingClassifier은 스케일링을 하지 않으면 에러가 발생한다.

#.4 평가, 예측
print(model.score(x_test, y_test))



# LinearRegression의 결과
# 0.5018716879686451

# XGBRegressor의 결과
# 0.9814814814814815