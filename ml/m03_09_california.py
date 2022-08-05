from unittest import result   
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()

x=datasets.data
y=datasets.target

# print(x)
# [[   8.3252       41.            6.98412698 ...    2.55555556
#     37.88       -122.23      ]
#  [   8.3014       21.            6.23813708 ...    2.10984183
#     37.86       -122.22      ]
#  [   7.2574       52.            8.28813559 ...    2.80225989
#     37.85       -122.24      ]
#  ...
#  [   1.7          17.            5.20554273 ...    2.3256351
#     39.43       -121.22      ]
#  [   1.8672       18.            5.32951289 ...    2.12320917
#     39.43       -121.32      ]'
#  [   2.3886       16.            5.25471698 ...    2.61698113
#     39.37       -121.24      ]]

# print(y)
# [4.526 3.585 3.521 ... 0.923 0.847 0.894]

print(x.shape, y.shape)   # x (20640, 8) 열이 8개   /  y (20640,)  

print(datasets.feature_names)  
 # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

print(datasets.DESCR)



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=68
                                                    )



#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression # LogisticRegression는 분류임  /  LinearRegression는 회귀임
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model_1 = LinearSVC()
model_2 = Perceptron()
model_3 = LinearRegression()
model_4 = KNeighborsRegressor()  
model_5 = DecisionTreeRegressor()
model_6 = RandomForestRegressor()


#2. 컴파일, 훈련
model_1.fit(x_train, y_train)
model_2.fit(x_train, y_train)
model_3.fit(x_train, y_train)
model_4.fit(x_train, y_train)
model_5.fit(x_train, y_train)
model_6.fit(x_train, y_train)



#4. 평가, 예측
from sklearn.metrics import r2_score
y_predict_1 = model_1.predict(x_test)
y_predict_2 = model_2.predict(x_test)
y_predict_3 = model_2.predict(x_test)
y_predict_4 = model_2.predict(x_test)
y_predict_5 = model_2.predict(x_test)
y_predict_6 = model_2.predict(x_test)

acc_1 = r2_score(y_test, y_predict_1)
acc_2 = r2_score(y_test, y_predict_2)
acc_3 = r2_score(y_test, y_predict_3)
acc_4 = r2_score(y_test, y_predict_4)
acc_5 = r2_score(y_test, y_predict_5)
acc_6 = r2_score(y_test, y_predict_6)
print('LinearSVC_accuracy : ', acc_1)
print('Perceptron_accuracy : ', acc_2)
print('LogisticRegression_accuracy : ', acc_3)
print('KNeighborsClassifier_accuracy : ', acc_4)
print('DecisionTreeClassifier_accuracy : ', acc_5)
print('RandomForestClassifier_accuracy : ', acc_6)

results_1 = model_1.score(x_test, y_test)
results_2 = model_2.score(x_test, y_test)
results_3 = model_2.score(x_test, y_test)
results_4 = model_2.score(x_test, y_test)
results_5 = model_2.score(x_test, y_test)
results_6 = model_2.score(x_test, y_test)
print("LinearSVC_결과 acc : ", results_1)   # 회기는 r2 / 분류는 acc로 결과가 나온다.
print("Perceptron_결과 acc : ", results_2)  
print("LogisticRegression_결과 acc : ", results_3)  
print("KNeighborsClassifier_결과 acc : ", results_4)  
print("DecisionTreeClassifier_결과 acc : ", results_5)  
print("RandomForestClassifier_결과 acc : ", results_6)  




