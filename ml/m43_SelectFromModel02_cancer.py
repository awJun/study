from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel

#1.데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) #(442, 10) (442,)


# import numpy as np
# x = np.delete(x, [2, 4, 7, 8, 14, 16, 21, 22, 23, 26, 27], axis=1)    # []안에 인덱스 번호를 입력 0~n
# # print(x.shape) 


# x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123) # , stratify=y

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# n_splits = 5
# kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 123)

# # 'n_estimators' : [100, 200, 300, 400, 500, 1000] # 디폴트 100 / 1~inf  (inf: 무한대)
# # 'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3/ 0~1 / eta라고 써도 먹힘
# # 'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~ inf / 정수
# # 'gamma': [0, 1, 2, 3, 4, 5, 7, 10, 100] 디폴트 0/ 0~inf
# # 'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10] 디폴트 1 / 0~inf
# # 'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# # 'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# # 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1

# # 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# # 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 0/ 0~inf / L1 절대값 가중치 규제 /alpha
# # 'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 1/ 0~inf/ L2 제곱 가중치 규제 /lambda

# parameters = {'n_estimators' : [100],
#               'learning_rate': [0.1],
#               'max_depth': [3],
#               'gamma': [1],
#               'min_child_weight': [1],
#               'subsample': [1],
#               'colsample_bytree': [1],
#               'colsample_bylevel': [1],
#               'colsample_bynode': [1] ,
#               'reg_alpha': [0],
#               'reg_lambda':[1]
#               }

# # https://xgboost.readthedocs.io/en/stable/parameter.html

# #2.모델 
# model = XGBRegressor(random_state=123,
#                       n_estimators=1000,
#                       learning_rate=0.1,
#                       max_depth=3,
#                       gamma=1,
#                     )

# # model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

# import time

# start_time = time.time()
# model.fit(x_train, y_train, # early_stopping_rounds=100,
#           # eval_set=[(x_train, y_train), (x_test, y_test)],
#           eval_metric='error', 
#           # 회귀 : rmse, mae, resle
#           # 이진 : error, auc...mlogloss...
#           # 다중이 : merror, mlogloss...
#           ) 
# # early_stopping_rounds 10번 동안 갱신이 없으면 정지시킨다
# # AssertionError: Must have at least 1 validation dataset for early stopping. 1개의 발리데이션 데이터가 필요하다
# # eval_set=[(x_train, y_train), (x_test, y_test)] 훈련하고 적용시킨다 이렇게 써도 가능
# # 얼리스타핑은 eval_set의 (x_test, y_test)에 적용시킨다 
# end_time = time.time() - start_time


# results = model.score(x_test, y_test)
# print('score :', results)  # 0.9736842105263158

# # y_predict = model.predict(x_test)
# # acc = r2_score(y_test, y_predict)
# # print('진짜 최종점수 test 점수 :', acc)

# print(model.feature_importances_)
# # [0.05553258 0.08159578 0.17970088 0.08489988 0.05190951 0.06678733
# #  0.05393131 0.08722917 0.27590865 0.06250493]

# thresholds = model.feature_importances_
# print('___________________________')
# for thresh in thresholds:
#     selection = SelectFromModel(model, threshold=thresh, prefit=True) # prefit 크거나 같은 컬럼을 빼준다
  
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     print(select_x_train.shape, select_x_test.shape)  
    
#     selection_model = XGBRegressor(n_jobs=-1,   
#                                    random_state=123,
#                                    n_estimators=100,
#                                    learning_rate=0.1,
#                                    max_depth=3,
#                                    gamma=1)
    
#     selection_model.fit(select_x_train, y_train)
    
#     y_predict = selection_model.predict(select_x_test)
#     score = r2_score(y_test, y_predict)
    
#     print("Thresh = %.3f, n=%d, R2: %.2f%% "
#           %(thresh, select_x_train.shape[1], score*100))
    
# print("걸린시간 : ",end_time)   
 
######[ 성능 분석 ]################################################################################################################################    
 
# [컬럼삭제 안하고]    
# score : 0.8826754911082894 
# 걸린시간 :  0.5952756404876709
 
# [14, 16 삭제]  # 원래 성능보다 더 좋은 값
# score : 0.8826754911082894         # 성능차이 없음
#  걸린시간 :  0.5682432651519775    

# [2, 4, 7, 8, 21, 22, 23, 26, 27 삭제]  원래 성능보다 안좋은 값 전부
# score : 0.8823677554366234         # 성능 더 안좋아짐
# 걸린시간 :  0.5947551727294922

# [2, 4, 7, 8, 14, 16, 21, 22, 23, 26, 27 삭제] 원래 성능보다 더 좋고 안좋은 값 전부  
# score : 0.8846861898078768
# 걸린시간 :  0.5217926502227783

#######[ 결과 ]###################################################################################################################################    

# cancer는 성능이 너무 좋은 컬럼과 안좋은 컬럼을 모두 없애니까 성능이 좋아졌다.

###################################################################################################################################       

# score : 0.8826754911082894

# Thresh = 0.000, n=30, R2: 88.27%    (성능 최대치) 1.
# (455, 9) (114, 9)
# Thresh = 0.025, n=9, R2: 88.22%                   2  
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%    (성능 최대치) 3.
# (455, 10) (114, 10)
# Thresh = 0.022, n=10, R2: 88.24%                  4 
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%    (성능 최대치) 5.
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%    (성능 최대치) 6.
# (455, 7) (114, 7)
# Thresh = 0.033, n=7, R2: 88.13%                   7
# (455, 2) (114, 2)
# Thresh = 0.289, n=2, R2: 83.11%                   8
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%    (성능 최대치) 9. 
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%    (성능 최대치) 10.
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%    (성능 최대치) 11.
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%    (성능 최대치) 12.
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%    (성능 최대치) 13.
# (455, 11) (114, 11)
# Thresh = 0.018, n=11, R2: 88.39%    (score 성능보다 더 좋은 컬럼)
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%    (성능 최대치) 15.
# (455, 12) (114, 12)
# Thresh = 0.016, n=12, R2: 88.39%    (score 성능보다 더 좋은 컬럼)
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%    (성능 최대치) 17.
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%    (성능 최대치) 18.
# (455, 13) (114, 13)
# Thresh = 0.016, n=13, R2: 88.27%    (성능 최대치) 19.
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%    (성능 최대치) 20.
# (455, 1) (114, 1)
# Thresh = 0.291, n=1, R2: 76.58% 
# (455, 6) (114, 6)
# Thresh = 0.042, n=6, R2: 87.63% 
# (455, 8) (114, 8)
# Thresh = 0.032, n=8, R2: 88.15% 
# (455, 3) (114, 3)
# Thresh = 0.108, n=3, R2: 83.23% 
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%     (성능 최대치) 24.
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%     (성능 최대치) 25.
# (455, 5) (114, 5)
# Thresh = 0.043, n=5, R2: 86.19% 
# (455, 4) (114, 4)
# Thresh = 0.066, n=4, R2: 87.68% 
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%      (성능 최대치) 28.
# (455, 30) (114, 30)
# Thresh = 0.000, n=30, R2: 88.27%      (성능 최대치) 29.

