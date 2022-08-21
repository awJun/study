
import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier      # pip install lightgbm
from catboost import CatBoostClassifier  # pip install catboost


#1. 데이터 
datasets = fetch_covtype()

# df = pd.DataFrame(datasets.data, columns=datasets.feature_names)   # x 데이터만 들어감
# print(df.head(7))

x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=123,
                                                    stratify=datasets.target
                                                    )

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

scaler = StandardScaler()
x_trian = scaler.fit_transform(x_train)
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


# LogisticRegression 사용
# 0.28075006669363095

# XGBClassifier 사용
# 0.233049060695507
