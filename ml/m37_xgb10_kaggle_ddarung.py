# Dacon 따릉이 문제풀이
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함


# 1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
# train_set = train_set.dropna() # nan 값(결측치) 열 없앰
train_set = train_set.fillna(0) # 결측치 0으로 채움
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

# trainset과 testset의 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler() 
scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)

parameters = {'n_estimators' : [100],
              'learning_rate' : [0.1],
              'max_depth' : [3],        # 디폴트 6  가지치기를 한다 ? 검색해보자 ..ㅠ   max가 깊어지면 과접합  /  낮게 잡을수록 좋다?
              'gamma' : [1],                     #  감마 알아서 찾아 ~
              'min_child_weight' : [1],
              'subsample' : [1],
              'colsample_bytree' : [1],
              'colsample_bylevel' : [1],
              'colsample_bynode' : [1],
              'reg_alpha' : [0],
              'reg_lamdba' : [0, 0.1, 0.01, 0.001, 1, 2, 10]
              }  
                  
#2. 모델구성
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import KFold
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=9)

xgb = XGBRegressor(random_state = 123)

model = GridSearchCV(xgb, parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)


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


# best_score_:  0.7723130201858387
# model.score:  0.7516087435613471
# acc score:  0.7516087435613471
# best tuned acc:  0.7516087435613471
# 걸린시간:  3.04 초