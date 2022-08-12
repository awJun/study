"""
[핵심]
feature_importances_를 사용해서 성능이 안좋은 컬럼을 확인하고 그 컬럼을 제거했을 때 성능이 어떻게
변하는지 확인하는 것이 목표이다.

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
################################################
# 02번을 가져와서 피쳐 한 개 삭제하고 성능 비교
################################################

import numpy as np
from sklearn.datasets import load_iris, load_diabetes

#1. 데이터
datasets = load_diabetes()
# print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# print(datasets['feature_names'])
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# columns = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x = datasets.data
y = datasets.target


# print(x)
# print(x.shape)    # (442, 10)
x = np.delete(x, 1, axis=1)  
# print(x.shape)    # (442, 9)



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


# r2_score :  0.15326487498122898
# DecisionTreeRegressor()  :  [0.09640383 0.01972624 0.23444866 0.05362851 0.04715042 0.05682229
#  0.04120031 0.01268737 0.36358732 0.07434506]

# r2_score :  0.5294850409206739
# RandomForestRegressor()  :  [0.05690817 0.01234359 0.30711507 0.09845901 0.04182044 0.05437573
#  0.05337027 0.0274514  0.27281351 0.07534281]

# r2_score :  0.5552730898249811
# GradientBoostingRegressor()  :  [0.04954857 0.01087554 0.30368709 0.11154877 0.02723317 0.05601129
#  0.04005201 0.01871274 0.33825881 0.04407199]

# r2_score :  0.4590400803596264
# XGBRegressor()  :   [0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 0.04843819
#  0.06012432 0.09595273 0.30483875 0.06629313]



##[ 컬럼 날린 후 결과 ]

# r2_score :  -0.09475064998282856
# DecisionTreeRegressor()  :  [0.0766278  0.24303727 0.05865016 0.05862545 0.04155146 0.03413404
#  0.027563   0.36846493 0.09134589]

# r2_score :  0.5093159988234124
# RandomForestRegressor()  :  [0.055939   0.30355532 0.10369613 0.04476883 0.05517733 0.05531655
#  0.02875577 0.26942041 0.08337065]

# r2_score :  0.5196994100045145
# GradientBoostingRegressor()  :  [0.04866268 0.3042278  0.11281764 0.02770715 0.05226841 0.03962066
#  0.02116233 0.33873736 0.05479597]

# r2_score :  0.2944509184380294
# XGBRegressor()  :   [0.04040969 0.17561655 0.08303527 0.04604991 0.05742574 0.0639853
#  0.10045296 0.34221125 0.09081327]

# 성능이 더 안좋아진걸 알 수 있엇다.. 튜닝 항목중 하나임 ㅋ

