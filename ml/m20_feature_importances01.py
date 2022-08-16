"""
[핵심]
<<<<<<< HEAD
feature_importances_를 사용해서 각각의 컬럼의 훈련 결과를 확인한다.
=======
feature_importances_를 사용해서 각각의 컬럼의 훈련 결과를 알려준다.
>>>>>>> ddaee9e4f1258d5dea9588ed5486a257fe6bbfd5

그 이후 x = np.delete(x, 1, axis=1)를 사용해서 원하는 컬럼을 삭제
 - x 데이터 안에 1이라는 인덱스에 위치에 있는 열을 삭제하겠다. 라는 의미



[ 데이터 안에 컬럼 확인 ]
datasets.feature_names 또는 datasets['feature_names']를 사용해서 컬럼을 확인한다.

# print(datasets.feature_names) 
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(datasets['feature_names'])
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']


[ 데이터 안에 컬럼 삭제 ]
x = np.delete(x, 1, axis=1)    # x 데이터안에 인덱스 1번의 위치한 열을 삭제하겠다 라는 뜻 
print(x.shape)      # 컬럼 삭제 후 shape를 찍어서 삭제가 되었는지 확인하는 과정


[ 성능이 안좋은 컬럼을 삭제하는 이유 ]
성능이 좋아질수도 있고 안좋아 질수도 있어서 튜닝 항목중 하나임


여기서는 컬럼을 확인만 했다.

"""


import numpy as np
from sklearn.datasets import load_iris

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier # pip install xgboost

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score: ', result)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc_score: ', acc)

print('----------------------------------------')
print(model, ': ', model.feature_importances_) # tree계열에만 있음

# DecisionTreeClassifier()     :  [0.         0.01669101 0.07659085 0.90671814] // 각 열별 중요도를 나타냄
# RandomForestClassifier()     :  [0.10766389 0.03133571 0.44818798 0.41281242]
# GradientBoostingClassifier() :  [0.00549151 0.01359283 0.30271053 0.67820512]
# XGBClassifier()              :  [0.00912187 0.0219429  0.678874   0.29006115]
