# https://xgboost.readthedocs.io/en/stable/parameter.html?highlight=subsample  xgb 공식문서

'''
[핵심]
얼리스타핑 사용방법

model.fit(x_train, y_train,
          early_stopping_rounds=10,   # 10번동안 갱신이 없으면 훈련을 종료하겠다.
          eval_set=[(x_test, y_test)],   # validation 설정해야지 얼리스탑핑이 걸린다.    / eval_set은 매트릭스와 같은거임
          eval_metric='error'   # 공식문서에서 사용가능 항목 확인 https://xgboost.readthedocs.io/en/stable/parameter.html?highlight=subsample
          # 회귀 : rmse, mae, rmsle...
          # 이진 : error, auc..., logloss...
          # 다중 : merror, mlogloss...      이런씩으로 공식문서보고 알아서 정리해놔 ~
          ) 

# AssertionError: Must have at least 1 validation dataset for early stopping. 1개의 발리데이션 데이터가 필요하다
# eval_set=[(x_train, y_train), (x_test, y_test)] 훈련하고 적용시킨다 이렇게 써도 가능 이렇게 사용할 경우 출력이 2개가 된다.
#  - 앞에는 (x_train, y_train)에 대한 로그값이고 뒤에는 앞에서 훈련되고 (x_test, y_test)에 적용된 로그값이 출력된다.
#  - 만약 (x_train, y_train)만 써서 사용할 경우 디폴트로 알아서 (x_train, y_train)가 들어가고 (x_train, y_train)의 로그값은 출력은 안된다.
#  - 즉! 사용하고 싶은 거 사용해도됨 둘 다 똑같은거임



[여기서는 파라미터를 안에 항목에 안넣엇으므로 모델 안에 따로 파라미터를 넣고 fit을 해야 파라미터를 넣어서 사용이 가능하다.]

 - 파라미터 넣는방법
model = XGBClassifier(random_state=123,
                      n_estimators=1000,
                      learning_rate=0.1,
                      max_depth=3,
                      gamma=1,
                     )

 - 이후 동일하게 훈련시키면 된다!
model.fit(x_train, y_train,
          early_stopping_rounds=10,   # 10번동안 갱신이 없으면 훈련을 종료하겠다.
          eval_set=[(x_test, y_test)],   # validation 설정해야지 얼리스탑핑이 걸린다.    / eval_set은 매트릭스와 같은거임
          eval_metric='error

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

# 베이지 서치? 이건 소수점까지 계산해주므로 훨씬 성능이 좋음


# parameters = {'n_estimators' : [100, 200, 300, 400, 500, 1000]}  
parameters = {'n_estimators' : [100],
              'learning_rate': [0.1],
              'max_depth': [3],
              'gamma': [1],
              'min_child_weight': [1],
              'subsample': [1],
              'colsample_bytree': [1],
              'colsample_bylevel': [1],
              'colsample_bynode': [1] ,
              'reg_alpha': [0],
              'reg_lambda':[1]
              }
              
              

#2. 모델
model = XGBClassifier(random_state=123,
                      n_estimators=1000,
                      learning_rate=0.1,
                      max_depth=3,
                      gamma=1,
                     )
              

# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(x_train, y_train,
          early_stopping_rounds=10,   # 10번동안 갱신이 없으면 훈련을 종료하겠다.
          eval_set=[(x_train, y_train), (x_test, y_test)],   # validation 설정해야지 얼리스탑핑이 걸린다.    / eval_set은 매트릭스와 같은거임
          eval_metric='error'   # 공식문서에서 사용가능 항목 확인 https://xgboost.readthedocs.io/en/stable/parameter.html?highlight=subsample
          # 회귀 : rmse, mae, rmsle...
          # 이진 : error, auc..., logloss...
          # 다중 : merror, mlogloss...      이런씩으로 공식문서보고 알아서 정리해놔 ~
          ) 

# AssertionError: Must have at least 1 validation dataset for early stopping. 1개의 발리데이션 데이터가 필요하다
# eval_set=[(x_train, y_train), (x_test, y_test)] 훈련하고 적용시킨다 이렇게 써도 가능 이렇게 사용할 경우 출력이 2개가 된다.
#  - 앞에는 (x_train, y_train)에 대한 로그값이고 뒤에는 앞에서 훈련되고 (x_test, y_test)에 적용된 로그값이 출력된다.
#  - 만약 (x_train, y_train)만 써서 사용할 경우 디폴트로 알아서 (x_train, y_train)가 들어가고 (x_train, y_train)의 로그값은 출력은 안된다.
#  - 즉! 사용하고 싶은 거 사용해도됨 둘 다 똑같은거임
 

results = model.score(x_test, y_test)
print('최종점수 :', results)  # 0.9736842105263158

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('진짜 최종점수 test 점수 :', acc)


# [0]     validation_0-error:0.36842
# [1]     validation_0-error:0.50000
# [2]     validation_0-error:0.49123
# [3]     validation_0-error:0.50000
# [4]     validation_0-error:0.49123
# [5]     validation_0-error:0.50000
# [6]     validation_0-error:0.50000
# [7]     validation_0-error:0.50000
# [8]     validation_0-error:0.49123
# [9]     validation_0-error:0.49123
# 최종점수 : 0.631578947368421
# 진짜 최종점수 test 점수 : 0.631578947368421