"""
[핵심]

#2. 모델
###[ 이 셋이 삼대장이여 ~ ]###########
xg = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier(verbose=0)
#######################################

위에서 삼대장을 불러와서 
model = VotingClassifier(estimators=[('xg', xg), ('lg', lg), ('cat', cat)],
                         voting='soft',   # hard도 있다.
                         )
로 해주고 m49_Voting1과 동일하게 하면된다.
                         
회귀같은 경우에는 위에서 voting='soft' 부분을 빼야지 돌아간다.

하지만 현재 출력을 할때 cat에서 나오는 verbose 때문에 출력한 결과를 보기 어려운 상황이므로 
verbose를 안뜨게하거나 cat의 순서를 제일 앞으로 이동시켜서 verbose를 결과값 위에 뜨게하면 해결된다.

### 1에서 사용한 이부분을 사용하면 Cat에서 나오는 verbose 때문에 나머지가 안보이는 문제점이있음######
# 그래서 cat의 순서를 [xg, lg, cat]에서 [cat, xg, lg] 순서를 변경하거나 verbose를 안뜨게하면 된다.

# verbose 안뜨게 하는 방법은 CatBoostClassifier(verbose=0)부분에서 verbose=0을 해주면 된다.







from sklearn.ensemble import VotingClassifier

lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)

model = VotingClassifier(estimators=[('LR', lr), ('KNN', knn)],
                         voting='soft'   # hard도 있다.
                         )

위에서 VotingClassifier을 설정하고 아래에서 for문으로 위에서 만든 모델 두개를 동시에 돌렸다.

classifiers = [lr, knn]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_predict)
    class_name = model2.__class__.__name__   # .__class__.__name__  모델의 이름을 출력해준다.
    print("{0} 정확도: {1:.4f}".format(class_name, score2))  
    
    # {0} : format 첫번째 위치에 있는 class_name를 출력해라
    # {1:.4f} : format 두번째 위치에 있는 score2를 소수 4번째 자리까지 출력해라



[TMI]
# VotingClassifier soft는 평균 / hard는 0 0 1이면 0 투표 방식임  1 1 0이면 1 
#  통상적으로 soft가 성능이 더 좋다.


print("보킹 결과 : ", round(score, 4))  # 소수 4번째 자리까지 출력이라는 뜻 / 결과 값을 소수 4번째 자리까지 반올림하겠다.

"""


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold,\
    HalvingRandomSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

encording_columns = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig',
                    'LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
                    'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual',
                    'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                    'BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual',
                    'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
                    'PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']

non_encording_columns = ['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond',
                         'YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtFinSF2',
                         'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF',
                         'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',
                         'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea',
                         'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
                         'MiscVal','MoSold','YrSold']



#1. 데이터
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv') # + 명령어는 문자를 앞문자와 더해줌  index_col=n n번째 컬럼을 인덱스로 인식
            
test_set = pd.read_csv(path + 'test.csv') # 예측에서 쓸거임  3

###################### IQR 이용해서 train_set에서 이상치나온 행 삭제########################
def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5 * IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
        
    return multiple_outliers
        
Outliers_to_drop = detect_outliers(train_set, 2, ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold'])


train_set.loc[Outliers_to_drop]


train_set = train_set.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
train_set.shape

print(train_set)

#################긁어온거####################################

num_strong_corr = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageCars',
                   'FullBath','YearBuilt','YearRemodAdd']

num_weak_corr = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1',
                 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'BsmtFullBath',
                 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                 'Fireplaces', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

catg_strong_corr = ['MSZoning', 'Neighborhood', 'Condition2', 'MasVnrType', 'ExterQual',
                    'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType']

catg_weak_corr = ['Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                  'LandSlope', 'Condition1',  'BldgType', 'HouseStyle', 'RoofStyle', 
                  'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'Foundation', 
                  'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 
                  'HeatingQC', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                  'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                  'SaleCondition' ]

cols_fillna = ['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu',
               'GarageQual','GarageCond','GarageFinish','GarageType', 'Electrical',
               'KitchenQual', 'SaleType', 'Functional', 'Exterior2nd', 'Exterior1st',
               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',
               'MSZoning', 'Utilities']

for col in cols_fillna : 
    train_set[col].fillna('None', inplace=True)
    test_set[col].fillna('None', inplace=True)

total = train_set.isnull().sum().sort_values(ascending=False)
percent = (train_set.isnull().sum()/train_set.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'])

train_set.fillna(train_set.mean(), inplace=True)
test_set.fillna(test_set.mean(), inplace=True)

total = train_set.isnull().sum().sort_values(ascending=False)
percent = (train_set.isnull().sum()/train_set.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(train_set.isnull().sum().sum(), test_set.isnull().sum().sum()) # 0 0 출력시 결측치 확인 끝

id_test = test_set['Id']

to_drop_num = num_weak_corr
to_drop_catg = catg_weak_corr

cols_to_drop = ['Id'] + to_drop_num + to_drop_catg

for df in [train_set, test_set] :
    df.drop(cols_to_drop, inplace=True, axis = 1)
    
# 'MSZoning'
msz_catg2 = ['RM', 'RH']
msz_catg3 = ['RL', 'FV'] 

# Neighborhood
nbhd_catg2 = ['Blmngtn', 'ClearCr', 'CollgCr', 'Crawfor', 'Gilbert', 'NWAmes', 'Somerst', 'Timber', 'Veenker']
nbhd_catg3 = ['NoRidge', 'NridgHt', 'StoneBr']

# Condition2
cond2_catg2 = ['Norm', 'RRAe']
cond2_catg3 = ['PosA', 'PosN'] 

# SaleType
SlTy_catg1 = ['Oth']
SlTy_catg3 = ['CWD']
SlTy_catg4 = ['New', 'Con']

for df in [train_set, test_set]:
    
    df['MSZ_num'] = 1  
    df.loc[(df['MSZoning'].isin(msz_catg2) ), 'MSZ_num'] = 2    
    df.loc[(df['MSZoning'].isin(msz_catg3) ), 'MSZ_num'] = 3        
    
    df['NbHd_num'] = 1       
    df.loc[(df['Neighborhood'].isin(nbhd_catg2) ), 'NbHd_num'] = 2    
    df.loc[(df['Neighborhood'].isin(nbhd_catg3) ), 'NbHd_num'] = 3    

    df['Cond2_num'] = 1       
    df.loc[(df['Condition2'].isin(cond2_catg2) ), 'Cond2_num'] = 2    
    df.loc[(df['Condition2'].isin(cond2_catg3) ), 'Cond2_num'] = 3    
    
    df['Mas_num'] = 1       
    df.loc[(df['MasVnrType'] == 'Stone' ), 'Mas_num'] = 2 
    
    df['ExtQ_num'] = 1       
    df.loc[(df['ExterQual'] == 'TA' ), 'ExtQ_num'] = 2     
    df.loc[(df['ExterQual'] == 'Gd' ), 'ExtQ_num'] = 3     
    df.loc[(df['ExterQual'] == 'Ex' ), 'ExtQ_num'] = 4     
   
    df['BsQ_num'] = 1          
    df.loc[(df['BsmtQual'] == 'Gd' ), 'BsQ_num'] = 2     
    df.loc[(df['BsmtQual'] == 'Ex' ), 'BsQ_num'] = 3     
 
    df['CA_num'] = 0          
    df.loc[(df['CentralAir'] == 'Y' ), 'CA_num'] = 1    

    df['Elc_num'] = 1       
    df.loc[(df['Electrical'] == 'SBrkr' ), 'Elc_num'] = 2 


    df['KiQ_num'] = 1       
    df.loc[(df['KitchenQual'] == 'TA' ), 'KiQ_num'] = 2     
    df.loc[(df['KitchenQual'] == 'Gd' ), 'KiQ_num'] = 3     
    df.loc[(df['KitchenQual'] == 'Ex' ), 'KiQ_num'] = 4      
    
    df['SlTy_num'] = 2       
    df.loc[(df['SaleType'].isin(SlTy_catg1) ), 'SlTy_num'] = 1  
    df.loc[(df['SaleType'].isin(SlTy_catg3) ), 'SlTy_num'] = 3  
    df.loc[(df['SaleType'].isin(SlTy_catg4) ), 'SlTy_num'] = 4 

train_set.drop(['MSZoning','Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True)
test_set.drop(['MSZoning', 'Neighborhood' , 'Condition2', 'MasVnrType', 'ExterQual', 'BsmtQual','CentralAir', 'Electrical', 'KitchenQual', 'SaleType', 'Cond2_num', 'Mas_num', 'CA_num', 'Elc_num', 'SlTy_num'], axis = 1, inplace = True)


##############################긁어온거끝################################

x = train_set.drop(['SalePrice'], axis=1)
y = train_set['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=99)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1234)

parameters = {
            'n_estimators':[100],
            'learning_rate':[1],
            'max_depth':[None,2,3,4,5,6,7,8,9,10],
            'gamma':[0],
            'min_child_weight':[1],
            'subsample':[1],
            'colsample_bytree':[0,0.1,0.2,0.3,0.5,0.7,1] ,
            'colsample_bylevel':[1],
            'colsample_bynode':[0,0.1,0.2,0.3,0.5,0.7,1],
            'alpha':[0,0.1,0.01,0.001,1,2,10],
            'lambda':[0,0.1,0.01,0.001,1,2,10]
              }  




#2. 모델
###[ 이 셋이 삼대장이여 ~ ]###########
from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor 
from catboost import CatBoostRegressor

xg = XGBRegressor()
lg = LGBMRegressor()
cat = CatBoostRegressor(verbose=0)
#######################################

# lr = LogisticRegression()
# knn = KNeighborsClassifier(n_neighbors=8)

from sklearn.ensemble import VotingRegressor
model = VotingRegressor(estimators=[('xg', xg), ('lg', lg), ('cat', cat)],
                        # voting='soft',   # hard도 있다.   회귀에선 빼야지 돌아가네요 ...
                         )

# VotingClassifier soft는 평균 / hard는 0 0 1이면 0 투표 방식임  1 1 0이면 1 
#  통상적으로 soft가 성능이 더 좋다.

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
from sklearn.metrics import r2_score

y_predict = model.predict(x_test)

score = r2_score(y_test, y_predict)

print("보킹 결과 : ", round(score, 4))  # 소수 4번째 자리까지 출력이라는 뜻 / 결과 값을 소수 4번째 자리까지 반올림하겠다.
# 보킹 결과 :  0.9737


classifiers = [cat, xg, lg]
for model2 in classifiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test, y_predict)
    class_name = model2.__class__.__name__   # .__class__.__name__  모델의 이름을 출력해준다.
    print("{0} 정확도: {1:.4f}".format(class_name, score2))  
    
    # {0} : format 첫번째 위치에 있는 class_name를 출력해라
    # {1:.4f} : format 두번째 위치에 있는 score2를 소수 4번째 자리까지 출력해라


# 보킹 결과 :  0.8646
# CatBoostRegressor 정확도: 0.8708
# XGBRegressor 정확도: 0.8442
# LGBMRegressor 정확도: 0.8561