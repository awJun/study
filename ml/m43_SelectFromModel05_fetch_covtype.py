import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor


# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape)
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234, train_size=0.8)

scaler = MinMaxScaler()
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


#4. 평가, 예측
from sklearn.metrics import r2_score
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
from sklearn.feature_selection import SelectFromModel
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


# 최종점수 : 0.621377269309255
# 진짜 최종점수 test 점수 : 0.621377269309255
# [0.0664307  0.01600967 0.00497092 0.01341994 0.00840821 0.01452159
#  0.00903617 0.01194589 0.00965915 0.0148403  0.14219667 0.03148515
#  0.09730344 0.00728085 0.0062075  0.00852034 0.00828462 0.00580541
#  0.01065347 0.00521819 0.         0.0009833  0.00091321 0.01502921
#  0.00853956 0.00226393 0.01344989 0.00934839 0.00031403 0.00323361
#  0.02099923 0.00074286 0.00106289 0.00514307 0.00351467 0.03582575
#  0.02863032 0.00846684 0.00092359 0.00207628 0.00126223 0.00067886
#  0.00251593 0.01677714 0.01575845 0.07745476 0.01279705 0.00370606
#  0.03004903 0.00217417 0.05018864 0.03647455 0.06335116 0.03315306]
# ___________________________
# (464809, 4) (116203, 4)
# Thresh = 0.066, n=4, R2: 37.78% 
# (464809, 15) (116203, 15)
# Thresh = 0.016, n=15, R2: 44.25% 
# (464809, 38) (116203, 38)
# Thresh = 0.005, n=38, R2: 46.81% 
# (464809, 21) (116203, 21)
# Thresh = 0.013, n=21, R2: 46.78% 
# (464809, 31) (116203, 31)
# Thresh = 0.008, n=31, R2: 46.69% 
# (464809, 19) (116203, 19)
# Thresh = 0.015, n=19, R2: 46.98% 
# (464809, 27) (116203, 27)
# Thresh = 0.009, n=27, R2: 47.15% 
# (464809, 23) (116203, 23)
# Thresh = 0.012, n=23, R2: 47.71% 
# (464809, 25) (116203, 25)
# Thresh = 0.010, n=25, R2: 47.60% 
# (464809, 18) (116203, 18)
# Thresh = 0.015, n=18, R2: 45.83% 
# (464809, 1) (116203, 1)
# Thresh = 0.142, n=1, R2: 4.26% 
# (464809, 10) (116203, 10)
# Thresh = 0.031, n=10, R2: 41.89% 
# (464809, 2) (116203, 2)
# Thresh = 0.097, n=2, R2: 6.86% 
# (464809, 33) (116203, 33)
# Thresh = 0.007, n=33, R2: 46.84% 
# (464809, 34) (116203, 34)
# Thresh = 0.006, n=34, R2: 46.84% 
# (464809, 29) (116203, 29)
# Thresh = 0.009, n=29, R2: 46.90% 
# (464809, 32) (116203, 32)
# Thresh = 0.008, n=32, R2: 46.84% 
# (464809, 35) (116203, 35)
# Thresh = 0.006, n=35, R2: 46.84% 
# (464809, 24) (116203, 24)
# Thresh = 0.011, n=24, R2: 47.68% 
# (464809, 36) (116203, 36)
# Thresh = 0.005, n=36, R2: 46.84% 
# (464809, 54) (116203, 54)
# Thresh = 0.000, n=54, R2: 46.80% 
# (464809, 48) (116203, 48)
# Thresh = 0.001, n=48, R2: 46.80% 
# (464809, 50) (116203, 50)
# Thresh = 0.001, n=50, R2: 46.80% 
# (464809, 17) (116203, 17)
# Thresh = 0.015, n=17, R2: 44.21% 
# (464809, 28) (116203, 28)
# Thresh = 0.009, n=28, R2: 47.13% 
# (464809, 43) (116203, 43)
# Thresh = 0.002, n=43, R2: 46.81% 
# (464809, 20) (116203, 20)
# Thresh = 0.013, n=20, R2: 47.51% 
# (464809, 26) (116203, 26)
# Thresh = 0.009, n=26, R2: 47.17% 
# (464809, 53) (116203, 53)
# Thresh = 0.000, n=53, R2: 46.80% 
# (464809, 41) (116203, 41)
# Thresh = 0.003, n=41, R2: 46.81% 
# (464809, 13) (116203, 13)
# Thresh = 0.021, n=13, R2: 43.00% 
# (464809, 51) (116203, 51)
# Thresh = 0.001, n=51, R2: 46.80% 
# (464809, 47) (116203, 47)
# Thresh = 0.001, n=47, R2: 46.80% 
# (464809, 37) (116203, 37)
# Thresh = 0.005, n=37, R2: 46.89% 
# (464809, 40) (116203, 40)
# Thresh = 0.004, n=40, R2: 46.81% 
# (464809, 8) (116203, 8)
# Thresh = 0.036, n=8, R2: 41.06% 
# (464809, 12) (116203, 12)
# Thresh = 0.029, n=12, R2: 42.86% 
# (464809, 30) (116203, 30)
# Thresh = 0.008, n=30, R2: 47.17% 
# (464809, 49) (116203, 49)
# Thresh = 0.001, n=49, R2: 46.80% 
# (464809, 45) (116203, 45)
# Thresh = 0.002, n=45, R2: 46.80% 
# (464809, 46) (116203, 46)
# Thresh = 0.001, n=46, R2: 46.80% 
# (464809, 52) (116203, 52)
# Thresh = 0.001, n=52, R2: 46.80% 
# (464809, 42) (116203, 42)
# Thresh = 0.003, n=42, R2: 46.81% 
# (464809, 14) (116203, 14)
# Thresh = 0.017, n=14, R2: 43.04% 
# (464809, 16) (116203, 16)
# Thresh = 0.016, n=16, R2: 44.09% 
# (464809, 3) (116203, 3)
# Thresh = 0.077, n=3, R2: 8.52% 
# (464809, 22) (116203, 22)
# Thresh = 0.013, n=22, R2: 46.97% 
# (464809, 39) (116203, 39)
# Thresh = 0.004, n=39, R2: 46.81% 
# (464809, 11) (116203, 11)
# Thresh = 0.030, n=11, R2: 42.73% 
# (464809, 44) (116203, 44)
# Thresh = 0.002, n=44, R2: 46.80% 
# (464809, 6) (116203, 6)
# Thresh = 0.050, n=6, R2: 40.19% 
# (464809, 7) (116203, 7)
# Thresh = 0.036, n=7, R2: 41.21% 
# (464809, 5) (116203, 5)
# Thresh = 0.063, n=5, R2: 39.92% 
# (464809, 9) (116203, 9)
# Thresh = 0.033, n=9, R2: 41.72% 
