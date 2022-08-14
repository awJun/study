"""
[핵심]
GridSearchCV에서 시간이 오래걸린다는 단점을 보안하기 위해서 RamdomSearch가 만들어졌다.
RamdomSearch는 파라미터를 랜덤으로 골라서 훈련시킨다. 
이로 인해서 연산량이 줄어들었으므로 훈련시간이 GridSearchCV에 비해서 줄어들었다.
GridSearchCV를 사용할 때 좋은 경우가 있고 RamdomSearch 좋은 경우가 있다. (튜닝 항목임)

[ RamdomSearch 사용법 ]
model = RamdomSearch(SVC(), parameters, cv=kfold, verbose=1,                # 42(parameters) * 5(kfold) = 210
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
   이것으로 인해서 m09에서 사용할 RamdomSearch가 나오게 됐다.

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

import pandas as pd 
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

# 1. 데이터
path = 'D:\study_data\_data\kaggle_bike/'
train_set = pd.read_csv(path+'train.csv')
# print(train_set)
# print(train_set.shape) # (10886, 11)

test_set = pd.read_csv(path+'test.csv')
# print(test_set)
# print(test_set.shape) # (6493, 8)

# datetime 열 내용을 각각 년월일시간날짜로 분리시켜 새 열들로 생성 후 원래 있던 datetime 열을 통째로 drop
train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # train_set에서 데이트타임 드랍
test_set.drop('datetime',axis=1,inplace=True) # test_set에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # casul 드랍 이유 모르겠음
train_set.drop('registered',axis=1,inplace=True) # registered 드랍 이유 모르겠음

#print(train_set.info())
# null값이 없으므로 결측치 삭제과정 생략

x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape) # (10886, 14) (10886,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
# scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)
parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'n_jobs':[-1]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10], 'n_jobs':[-1]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10], 'n_jobs':[-1]},
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
    ]    

#2. 모델구성
from sklearn.ensemble import RandomForestRegressor
# model = SVC(C=1, kernel='linear', degree=3)
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)

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
print('acc score: ', r2_score(y_test, ypred))
ypred_best = model.best_estimator_.predict(x_test)
print('best tuned acc: ', r2_score(y_test, ypred_best))

print('걸린시간: ', round(end-start,2), '초')

# # 5. 제출 준비
# submission = pd.read_csv(path + 'submission.csv', index_col=0)
# y_submit = model.predict(test_set)
# submission['count'] = np.abs(y_submit) # 마이너스 나오는거 절대값 처리

# submission.to_csv(path + 'submission.csv', index=True)

# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, min_samples_split=5, n_jobs=-1)
# 최적의 파라미터:  {'n_jobs': -1, 'min_samples_split': 5, 'min_samples_leaf': 3}
# best_score_:  0.941835537881442
# model.score:  0.9458354337017375
# acc score:  0.9458354337017375
# best tuned acc:  0.9458354337017375
# 걸린시간:  9.71 초
