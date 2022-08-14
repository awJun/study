'''
[핵심]


'''

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import r2_score, accuracy_score

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    shuffle=True,
                                                    random_state=123,
                                                    train_size=0.8,
                                                    stratify=y
                                                    )

scaler = MinMaxScaler()
x_trian = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)     
              

# #2. 모델
# model = XGBClassifier(random_state=123,
#                       n_estimators=1000,
#                       learning_rate=0.1,
#                       max_depth=3,
#                       gamma=1,                             # 요거 이후 갱신 읍다.
#                      )
#3. 훈련
# model.fit(x_train, y_train,
#           early_stopping_rounds=200,   # 10번동안 갱신이 없으면 훈련을 종료하겠다.
#           eval_set=[(x_train, y_train), (x_test, y_test)],   # validation 설정해야지 얼리스탑핑이 걸린다.    / eval_set은 매트릭스와 같은거임
#           eval_metric='error'   
#           )


# import pickle
import joblib
path = 'd:/study_data/_save/_xg/'
model = joblib.load(path + 'm39_joblib1_save.dat')  


# 4. 평가, 예측
print('테스트 스코어: ', model.score(x_test, y_test))

acc = accuracy_score(y_test, model.predict(x_test))
print('acc_score 결과: ', acc)

