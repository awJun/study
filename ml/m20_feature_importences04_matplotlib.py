"""
[핵심]
feature_importances_를 사용해서 성능이 안좋은 컬럼을 확인하고 그 컬럼을 제거했을 때 성능이 어떻게
변하는지 확인하는 것이 목표이다.

"""

import numpy as np
from sklearn.datasets import load_iris, load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                     train_size=0.8,
                                                     shuffle=True,
                                                     random_state=123
                                                     )


#2. 모델구성
from sklearn.tree import DecisionTreeRegressor # 나무 ~
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor     # Regressor : 회기 / Classifier 분류
from xgboost import XGBClassifier, XGBRegressor  # xgboost 깔아야 사용 가능하다.


# model = DecisionTreeRegressor() 
model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = XGBRegressor()
#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print("model.score : ", result)

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print("r2_score : ", acc)

print("==========================")
# print(model," : ", model.feature_importances_)  # feature_importances_는 트리에만 있는거임
                                   # 전체 픽쳐중 성능이 안좋은 것은 빼도 되는지 확인하는 용도임 acc와 같이  0 ~ 1의 값을 보여줌

import matplotlib.pyplot as plt

def plot_feature_importances(model):
    n_features = datasets.data.shape[1]  # data.shape[1] 열을 사용하겠다
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("features")
    plt.ylim(-1, n_features)
    
plot_feature_importances(model)
plt.show()

# DecisionTreeRegressor()  :  [0.10190659 0.02310902 0.23213198 0.05170539 0.04256703 0.04765942
#  0.03820782 0.0250506  0.36488463 0.07277753]

# RandomForestRegressor()  :  [0.05612309 0.01273293 0.29057427 0.10423922 0.0405132  0.05143917
#  0.05585144 0.02677097 0.27763685 0.08411887]

# GradientBoostingRegressor()  :  [0.04965624 0.01086936 0.30371012 0.11159881 0.02890053 0.05450568
#  0.03974847 0.01840977 0.33881486 0.04378616]

# r2_score :  0.4590400803596264
# XGBRegressor()  :   [0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 0.04843819
#  0.06012432 0.09595273 0.30483875 0.06629313]












