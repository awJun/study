"""
[핵심]
feature_importances_를 사용해서 각각의 컬럼의 훈련 결과를 확인한다.

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
import matplotlib.pyplot as plt

def plot_feature_importances(model): # 그림 함수 정의
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
                # x                     y
    plt.yticks(np.arange(n_features), datasets.feature_names) # 눈금 설정
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features) # ylimit : 축의 한계치 설정

models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]
print(str(models[3]))

# 3. 훈련
plt.figure(figsize=(10,5))
for i in range(len(models)):
    models[i].fit(x_train, y_train)
    plt.subplot(2,2, i+1)
    plot_feature_importances(models[i])
    if str(models[i]).startswith('XGBClassifier'):
        plt.title('XGB()')
    else:
        plt.title(models[i])

# plt.subplot(2,2,1)
# plot_feature_importances(model1)

# plt.subplot(2,2,2)
# plot_feature_importances(model2)

# plt.subplot(2,2,3)
# plot_feature_importances(model3)

# plt.subplot(2,2,4)
# plot_feature_importances(model3)

plt.show()


# model.score:  -0.06803668834514842
# r2_score:  -0.06803668834514842
# DecisionTreeClassifier() :  [0.08175505 0.01452838 0.34378855 0.08707457 0.02062155 0.10143098
#  0.06139199 0.01179111 0.15634859 0.12126924]

# model.score:  0.4076785129628565
# r2_score:  0.4076785129628565
# RandomForestClassifier() :  [0.05768957 0.01261572 0.33354815 0.09110822 0.04410219 0.06195914
#  0.06235012 0.02609681 0.22340702 0.08712305]

# model.score:  0.4124988763421431
# r2_score:  0.4124988763421431
# GradientBoostingClassifier() :  [0.04612784 0.01648727 0.33594916 0.0955424  0.03161077 0.06604381
#  0.03821368 0.01413885 0.27693126 0.07895497]

# model.score:  0.26078151031491137
# r2_score:  0.26078151031491137
# XGBClassifier :  [0.02666356 0.06500483 0.28107476 0.05493598 0.04213588 0.0620191 0.06551369 0.17944618 0.13779876 0.08540721]

