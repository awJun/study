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
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.metrics import accuracy_score, r2_score

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(n_estimators=1000,
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
              tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234
              )

model.fit(x_train, y_train, early_stopping_rounds=10, 
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric=['logloss'])

print('테스트 스코어: ', model.score(x_test, y_test))

print('-----------------------------------------------------------------------')
hist = model.evals_result()
print(hist)

# [실습] 그래프 그리기
import matplotlib.pyplot as plt
# plt.subplot(2,1,1)
# plt.plot(hist['validation_0']['logloss'])

# plt.subplot(2,1,2)
# plt.plot(hist['validation_1']['logloss'])

plt.figure(figsize=(10,10))
for i in range(len(hist.keys())):
    plt.subplot(len(hist.keys()),1, i+1)
    plt.plot(hist['validation_'+str(i)]['logloss'])
    plt.xlabel('n_estimators')
    plt.ylabel('evals_result')
    plt.title('validation_'+str(i))

plt.show()











# 최상의 매개변수:  {'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.5, 'gamma': 0, 'learning_rate': 1, 'max_depth': 2, 'min_child_weight': 1, 'n_estimators': 100, 'reg_alpha': 0.01, 'reg_lambda': 1, 'subsample': 1}
# 최상의 점수:  0.9824175824175825
# 테스트 스코어:  0.9649122807017544

# n_estimators = 100
# 테스트 스코어:  0.9912280701754386