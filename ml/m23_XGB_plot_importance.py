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
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()
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
### [ xgboost.plotting를 사용할 것이므로 주석처리 ] 
# def plot_feature_importances(model):
#     n_features = datasets.data.shape[1]  # data.shape[1] 열을 사용하겠다
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("features")
#     plt.ylim(-1, n_features)
    
# plot_feature_importances(model)
# plt.show()

##[ 메인 ]############################################################################
### [ xgboost.plotting를 사용할 것이므로 위에 plt는 주석처리 ] 
from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()

#판다스 or 넘파이의 따라 그래프가 다르게 찍힌다.
#######################################################################################













