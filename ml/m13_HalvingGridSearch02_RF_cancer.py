"""
[핵심]
 - GridSearchCV에서 시간이 오래걸린다는 단점을 보안하기 위해서 HalvingGridSearch도 만들어졌다.
 - HalvingGridSearch는 훈련을 시킬 때 마다 성능이 좋게나오게 하는 파라미터들을 전체에서 50%만 남기고
   남긴 부분으로만 훈련을 시킨다. 그 후 다시 훈련중 성능이 좋게 나오게하는 50%만 남기고 남긴 부분으로만
   훈련을 시킨다. 
이로 인해서 연산량이 줄어들었으므로 훈련시간이 GridSearchCV에 비해서 줄어들었다.
GridSearchCV를 사용할 때 좋은 경우가 있고 HalvingGridSearch 좋은 경우가 있다. (튜닝 항목임)

[ HalvingGridSearch 사용법 ]
model = HalvingGridSearch(SVC(), parameters, cv=kfold, verbose=1,        # 42(parameters) * 5(kfold) = 210
                     refit=True, n_jobs=1)  

refit=True : 최적의 파라미터를 찾은 후 그것을 다시 훈련시킬 것인가
            만약 Flse로 지정하면 아래에서 model.best_estimator_를 빼야 에러가 발생하지 않는다.

 n_jobs=1 : 컴퓨터의 cpu에서 얼마나 사용할 것인가에 대헤서 설정하는 부분 -1로하면 자동으로 전체를 사용한다.

print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=10, kernel='linear')
###############################################################[성능에 좌우]
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 10, 'degree': 3, 'kernel': 'linear'}
print("best_score_ : ", model.best_score_)
# best_score_ :  0.975
###################################################################
 

[ GridSearchCV의 문제점 ]
 - 시간이 오래걸린다.
   이것으로 인해서 m09에서 사용할 HalvingGridSearch도 나오게 됐다.

[ 파라미터 설명 ]
parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3, 4, 5]},      # 12
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},          # 6
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],                          # 24
    "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]}               
]                                                                           # 총 42번
"C":[1, 10, 100, 1000] 4개 / degree":[3, 4, 5] 3개  /  4 * 3 = 12
"C":[1, 10, 100] 3개 / "gamma":[0.001, 0.0001] 2개  / 3 * 2 = 6
"C":[1, 10, 100, 1000] / "gamma":[0.01, 0.001, 0.0001] / "degree":[3, 4]  /  4 * 3 * 2 = 24           

도합 연산냥 42번이다.  
여기서 훈련량에 따라서 "훈련량 # 42 = 총 연산량" 형태가 되므로 참고할 것
   
"""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score,\
    GridSearchCV, HalvingGridSearchCV

# 1. 데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=9)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'n_jobs':[-1,2,4]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10], 'n_jobs':[-1,2,4]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10], 'n_jobs':[-1,2,4]},
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
    ]                                                                                       
                      
#2. 모델구성
from sklearn.ensemble import RandomForestClassifier
model = HalvingGridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)

# 3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 매개변수: ', model.best_estimator_)
print('최적의 파라미터: ', model.best_params_)
print('best_score_: ', model.best_score_)
print('model.score: ', model.score(x_test, y_test))
ypred = model.predict(x_test)
print('acc score: ', accuracy_score(y_test, ypred))
ypred_best = model.best_estimator_.predict(x_test)
print('best tuned acc: ', accuracy_score(y_test, ypred_best))

print('걸린시간: ', round(end-start,2), '초')

# n_iterations: 3
# n_required_iterations: 5
# n_possible_iterations: 3
# min_resources_: 20
# max_resources_: 455
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 152
# n_resources: 20
# Fitting 5 folds for each of 152 candidates, totalling 760 fits
# ----------
# iter: 1
# n_candidates: 51
# n_resources: 60
# Fitting 5 folds for each of 51 candidates, totalling 255 fits
# ----------
# iter: 2
# n_candidates: 17
# n_resources: 180
# Fitting 5 folds for each of 17 candidates, totalling 85 fits
# 최적의 매개변수:  RandomForestClassifier(max_depth=8, n_jobs=2)
# 최적의 파라미터:  {'max_depth': 8, 'n_estimators': 100, 'n_jobs': 2}
# best_score_:  0.9666666666666666
# model.score:  0.9736842105263158
# acc score:  0.9736842105263158
# best tuned acc:  0.9736842105263158
# 걸린시간:  29.58 초