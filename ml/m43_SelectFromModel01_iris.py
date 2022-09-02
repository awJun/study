from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel

#1.데이터 
datasets = load_iris()
x = datasets.data
y = datasets.target 
# print(x.shape, y.shape) #(442, 10) (442,)


import numpy as np
x = np.delete(x, [2], axis=1)    # []안에 인덱스 번호를 입력 0~n
# print(x.shape) 


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123) # , stratify=y

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 123)

# 'n_estimators' : [100, 200, 300, 400, 500, 1000] # 디폴트 100 / 1~inf  (inf: 무한대)
# 'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3/ 0~1 / eta라고 써도 먹힘
# 'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~ inf / 정수
# 'gamma': [0, 1, 2, 3, 4, 5, 7, 10, 100] 디폴트 0/ 0~inf
# 'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10] 디폴트 1 / 0~inf
# 'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1

# 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 0/ 0~inf / L1 절대값 가중치 규제 /alpha
# 'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 1/ 0~inf/ L2 제곱 가중치 규제 /lambda

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

# https://xgboost.readthedocs.io/en/stable/parameter.html

#2.모델 
model = XGBRegressor(random_state=123,
                      n_estimators=1000,
                      learning_rate=0.1,
                      max_depth=3,
                      gamma=1,
                    )

# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

import time

start_time = time.time()
model.fit(x_train, y_train, # early_stopping_rounds=100,
          # eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric='error', 
          # 회귀 : rmse, mae, resle
          # 이진 : error, auc...mlogloss...
          # 다중이 : merror, mlogloss...
          ) 
# early_stopping_rounds 10번 동안 갱신이 없으면 정지시킨다
# AssertionError: Must have at least 1 validation dataset for early stopping. 1개의 발리데이션 데이터가 필요하다
# eval_set=[(x_train, y_train), (x_test, y_test)] 훈련하고 적용시킨다 이렇게 써도 가능
# 얼리스타핑은 eval_set의 (x_test, y_test)에 적용시킨다 
end_time = time.time() - start_time


results = model.score(x_test, y_test)
print('score :', results)  # 0.9736842105263158

# y_predict = model.predict(x_test)
# acc = r2_score(y_test, y_predict)
# print('진짜 최종점수 test 점수 :', acc)

print(model.feature_importances_)
# [0.05553258 0.08159578 0.17970088 0.08489988 0.05190951 0.06678733
#  0.05393131 0.08722917 0.27590865 0.06250493]

thresholds = model.feature_importances_
print('___________________________')
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) # prefit 크거나 같은 컬럼을 빼준다
  
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)  
    
    selection_model = XGBRegressor(n_jobs=-1,   
                                   random_state=123,
                                   n_estimators=100,
                                   learning_rate=0.1,
                                   max_depth=3,
                                   gamma=1)
    
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    
    print("Thresh = %.3f, n=%d, R2: %.2f%% "   # %.2f%% : 소수점 2번째 자리까지 출력해라라는 뜻 
          %(thresh, select_x_train.shape[1], score*100))
    
    
print("걸린시간 : ",end_time)
    
    
######[ 성능 분석 ]################################################################################################################################    
 
# [컬럼삭제 안하고]    
# score : 0.9497811046395627   
# 걸린시간 :  0.35045790672302246
 
# [3번째 컬럼삭제]    
# score : 0.9431581496426672
# 걸린시간 :  0.3728170394897461
    
#######[ 결과 ]###################################################################################################################################    

# iris는 컬럼을 삭제하니까 오히려 더 성능이 안좋아짐

###################################################################################################################################    
# score : 0.9497811046395627  
    
# (120, 4) (30, 4)
# Thresh = 0.000, n=4, R2: 94.98%    <-- 이 수치와 위에 score 스코어와 비교해서 위에 score 스코어랑 같을수록 해당 컬럼이 가장 성능이 좋은 뜻이다.  
# (120, 4) (30, 4)                         - 이것을 참고해서 성능이 구린 것을 삭제해보고 성능이 올라가는지 판단해볼 것
# Thresh = 0.000, n=4, R2: 94.98%    <-- 이 수치는 해당 컬럼을 뺏을 때 이정도 성능이 올라갈 수 있다라고 %로 알려주는거임
# (120, 1) (30, 1) 
# Thresh = 0.668, n=1, R2: 92.56%    <-- 이 수치는 해당 컬럼을 뺏을 때 이정도 성능이 올라갈 수 있다라고 %로 알려주는거임
# (120, 2) (30, 2)
# Thresh = 0.332, n=2, R2: 94.98%    <-- 이 수치는 해당 컬럼을 뺏을 때 이정도 성능이 올라갈 수 있다라고 %로 알려주는거임
    
    
    