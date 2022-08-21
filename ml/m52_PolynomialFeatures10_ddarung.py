
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


"""
[핵심]
PolynomialFeatures  : 단항식을 다항식으로 증폭시켜준다.

(원래 형태)
# # print(x)
# # [[0 1]
# #  [2 3]
# #  [4 5]
# #  [6 7]]

# # print(x.shape)   # (4, 2)

(사용법)
from sklearn.preprocessing import PolynomialFeatures
# pf = PolynomialFeatures(degree=2)  

# x_pf = pf.fit_transform(x)

# print(x_pf)
# # [[ 1.  0.  1.  0.  0.  1.]
# #  [ 1.  2.  3.  4.  6.  9.]  
# #  [ 1.  4.  5. 16. 20. 25.]
# #  [ 1.  6.  7. 36. 42. 49.]]
# print(x_pf.shape)   # (4, 6)


(증폭할 때 특징)
 - 증폭할 때 앞에 1은 무조건 채워준다.
 - 증폭할 때 PolynomialFeatures(degree=2)부분에  degree는 통상 2까지 넣는다. 
   3이상부터는 과접합으로 인해 성능이 저하될 가능성이 높기 때문에 잘 안넣어서 사용한다.
   
   
(증폭 연산 방법)
[ 1.  2.  3.  4.  6.  9.]    # 1채우고 / 원래 데이터 2 / 원래데이터 3 / 2^2 / 3*2 / 3^2
"""


#2. 모델
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
model = make_pipeline(StandardScaler(),
                      LogisticRegression())

model.fit(x_train, y_train)

print("그냥 스코어 : ", model.score(x_test, y_test))
# 0.7665382927362877

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
print("CV : ", scores)
print("CV 엔빵 : ", np.mean(scores))

############################### PolynomialFeatures 후 #######################################
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)   # include_bias=Falsed 음수로 나오는 것을 방지해줫다..? 정확히는 모르겟음
xp =  pf.fit_transform(x)

# print(xp.shape, )

#2. 모델
model = make_pipeline(StandardScaler(),
                      LinearRegression())



x_train, x_test, y_train, y_test = train_test_split(xp, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=1234,
                                                    )

model.fit(x_train, y_train)

print("PolynomialFeatures 후 스코어 : ", model.score(x_test, y_test))


# 증폭 전
# 0.7665382927362877

# 증폭 후
# 0.8745129304823926         # 데이터가 좋아서 과적합이 된 것이라고 하심

# 증폭한 데이터의 과적합 정도를 확인
from sklearn.model_selection import cross_val_score  # 스코어가 얼마나 정확한지 검증하는 용도
scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
print("폴리 CV : ", scores)
print("폴리 CV 엔빵 : ", np.mean(scores))


# 그냥 스코어 :  0.6030858858241663
# CV :  [0.56720231 0.63020751 0.55013821 0.61269437 0.57358677]
# CV 엔빵 :  0.5867658351643988

# PolynomialFeatures 후 스코어 :  0.5265025284862332
# 폴리 CV :  [0.62801538 0.61651215 0.61811896 0.58189259 0.6480483 ]
# 폴리 CV 엔빵 :  0.6185174765801309