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

import numpy as np
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.experimental import enable_halving_search_cv   # HalvingRandomSearchCV를 사용하기 위해 import함 
   # 무조건HalvingRandomSearchCV import보다 위에서 import 해야한다.
from sklearn.model_selection  import HalvingRandomSearchCV
from sklearn.metrics import accuracy_score
                        
#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)  #행(Instances): 150   /   열(Attributes): 4
# print(datasets.feature_names)

x = datasets['data']  # .data와 동일 
y = datasets['target']  
# print(x.shape)   # (150, 4)
# print(y.shape)   # (150,)
# print("y의 라벨값 : ", np.unique(y))  # 해당 데이터의 고유값을 출력해준다.


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

n_splits=5
kfold = KFold(n_splits=5, shuffle=True, random_state=101)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3, 4, 5]},      # 12
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},          # 6
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],                          # 24
    "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]}
]                                                                           # 총 42번


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression는 분류임
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = SVC(C=1, kernel='linear', degree=3)
model = HalvingRandomSearchCV(SVC(), parameters, cv=kfold, verbose=1,                # 42(parameters) * 5(kfold) = 210
                     refit=True, n_jobs=1)                                  # n_jobs 코어 갯수


#3. 컴파일, 훈련
import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()  -start_time

print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=10, kernel='linear')
###############################################################[성능에 좌우]
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 10, 'degree': 3, 'kernel': 'linear'}
print("best_score_ : ", model.best_score_)
# best_score_ :  0.975
################################################################
print("model.score : ", model.score(x_test, y_test))
# model.score :  0.9666666666666667

#4. 평가
y_predict = model.predict(x_test)
print("accuracy_score", accuracy_score(y_test, y_predict))
# accuracy_score 0.9666666666666667

# y_pred_best = model.best_estimator_.__prepare__(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))

print("걸린시간 : ", round(end_time, 2), "초")
# 걸린시간 :  0.24 초

# 통상적으로 컴퓨터가 좋으면 딥러닝 / 안좋으면 머신러닝을 활용한다.





