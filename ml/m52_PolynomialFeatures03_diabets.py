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

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

datasets = load_diabetes()
x, y = datasets.data, datasets.target
# print(x.shape, y.shape)    # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=1234,
                                                    )

#2. 모델
model = make_pipeline(StandardScaler(),
                      LinearRegression())

model.fit(x_train, y_train)

print("그냥 스코어 : ", model.score(x_test, y_test))
# 0.7665382927362877

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
print("CV : ", scores)
print("CV 엔빵 : ", np.mean(scores))

############################### PolynomialFeatures 후 #######################################

pf = PolynomialFeatures(degree=2, include_bias=False)   # include_bias=Falsed 음수로 나오는 것을 방지해줫다..? 정확히는 모르겟음
xp =  pf.fit_transform(x)

print(xp.shape, )   # (506, 105)

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

# 그냥 스코어 :  0.46263830098374936
# CV :  [0.51419232 0.36255395 0.41211099 0.65982774 0.48235245]
# CV 엔빵 :  0.486207490675583

# PolynomialFeatures 후 스코어 :  0.4186731903894866
# 폴리 CV :  [0.25778745 0.18665768 0.21776293 0.40354262 0.37293825]
# 폴리 CV 엔빵 :  0.28773778201217637