"""
[핵심]
feature_importances_를 사용해서 성능이 안좋은 컬럼을 확인하고 그 컬럼을 제거했을 때 성능이 어떻게
변하는지 확인하는 것이 목표이다.

여기서는 컬럼을 확인만 했다.

"""

import numpy as np
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                     train_size=0.8,
                                                     shuffle=True,
                                                     random_state=123
                                                     )


#2. 모델구성
from sklearn.tree import DecisionTreeClassifier  # 나무 ~
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier  # xgboost 깔아야 사용 가능하다.


# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()
#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print("model.score : ", result)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

print("==========================")
print(model," : ", model.feature_importances_)  
                                   # 전체 픽쳐중 성능이 안좋은 것은 빼도 되는지 확인하는 용도임 acc와 같이  0 ~ 1의 값을 보여줌



# DecisionTreeClassifier : [0.01253395 0.01253395 0.06761888 0.90731323]                                    
# RandomForestClassifier : [0.11383432 0.03147701 0.42821969 0.42646898]                                   
# GradientBoostingClassifier : [0.00080481 0.0243305  0.66505227 0.30981242]
# XGBClassifier : [0.0089478  0.01652037 0.75273126 0.22180054]


















