from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel

#1.데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) #(442, 10) (442,)

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
print('최종점수 :', results)  # 0.9736842105263158

y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print('진짜 최종점수 test 점수 :', acc)

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
    
    print("Thresh = %.3f, n=%d, R2: %.2f%% "
          %(thresh, select_x_train.shape[1], score*100))

print("걸린시간 : ",end_time) 

######[ 성능 분석 ]################################################################################################################################    
 
# [컬럼삭제 안하고]    
# score : 0.4378510933146712
# 걸린시간 :  0.37661242485046387
 


#######[ 결과 ]###################################################################################################################################    

# cancer는 성능이 너무 좋은 컬럼과 안좋은 컬럼을 모두 없애니까 성능이 좋아졌다.

###################################################################################################################################       
    

# score : 0.4378510933146712

# Thresh = 0.056, n=8, R2: 50.31%     # n은 컬럼의 갯수
# (353, 5) (89, 5)
# Thresh = 0.082, n=5, R2: 53.86% 
# (353, 2) (89, 2)
# Thresh = 0.180, n=2, R2: 52.62% 
# (353, 4) (89, 4)
# Thresh = 0.085, n=4, R2: 51.18% 
# (353, 10) (89, 10)
# Thresh = 0.052, n=10, R2: 51.19% 
# (353, 6) (89, 6)
# Thresh = 0.067, n=6, R2: 53.41%
# (353, 9) (89, 9)
# Thresh = 0.054, n=9, R2: 51.88%
# (353, 3) (89, 3)
# Thresh = 0.087, n=3, R2: 50.44%
# (353, 1) (89, 1)
# Thresh = 0.276, n=1, R2: 32.91%
# (353, 7) (89, 7)
# Thresh = 0.063, n=7, R2: 54.05%   

