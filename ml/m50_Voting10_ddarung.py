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


# Dacon 따릉이 문제풀이
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold,\
    HalvingRandomSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)




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



# 보킹 결과 :  0.7987
# CatBoostRegressor 정확도: 0.7997
# XGBRegressor 정확도: 0.7679
# LGBMRegressor 정확도: 0.7849





