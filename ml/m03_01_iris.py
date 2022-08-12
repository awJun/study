"""
[핵심]
각 모델 성능 테스트  (해당 모델들은 분류 모델들이다.)
model_1 = LinearSVC()
model_2 = Perceptron()
model_3 = LogisticRegression()
model_4 = KNeighborsClassifier()
model_5 = DecisionTreeClassifier()
model_6 = RandomForestClassifier()

neighbors, tree, ensemble 모델 정리
from sklearn.neighbors   # 이웃하는 ..? 검색하자
from sklearn.tree        # 더 좋은 것을 아래로 뿌리를 내린다(가지치기) 결정나무
from sklearn.ensemble    # 같이 넣었을 때 더 좋은 것을 캐치

[중요]
LogisticRegression
- 이것은 Regression들어가지만 분류 모델이다.
- LinearRegression 이친구가 회기 모델이다.
이 친구 빼고는 나머지는 다 Regression이 들어가면 회기 모델로 생각하면 된다.

Classifier가 들어가면 분류 모델로 생각하면 된다.

"""

import numpy as np
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
                        
#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)  #행(Instances): 150   /   열(Attributes): 4
# print(datasets.feature_names)

x = datasets['data']  # .data와 동일 
y = datasets['target']  
# print(x.shape)   # (150, 4)
# print(y.shape)   # (150,)
# print("y의 라벨값 : ", np.unique(y))  # 해당 데이터의 고유값을 출력해준다.


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

#2. 모델구성
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier   # 이웃하는 ..? 검색하자
from sklearn.tree import DecisionTreeClassifier      # 더 좋은 것을 아래로 뿌리를 내린다(가지치기) 결정나무
from sklearn.ensemble import RandomForestClassifier  # 같이 넣었을 때 더 좋은 것을 캐치

model_1 = LinearSVC()
model_2 = Perceptron()
model_3 = LogisticRegression()
model_4 = KNeighborsClassifier()
model_5 = DecisionTreeClassifier()
model_6 = RandomForestClassifier()


#2. 컴파일, 훈련
model_1.fit(x_train, y_train)
model_2.fit(x_train, y_train)
model_3.fit(x_train, y_train)
model_4.fit(x_train, y_train)
model_5.fit(x_train, y_train)
model_6.fit(x_train, y_train)


#4. 평가, 예측
from sklearn.metrics import accuracy_score
y_predict_1 = model_1.predict(x_test)
y_predict_2 = model_2.predict(x_test)
y_predict_3 = model_2.predict(x_test)
y_predict_4 = model_2.predict(x_test)
y_predict_5 = model_2.predict(x_test)
y_predict_6 = model_2.predict(x_test)

acc_1 = accuracy_score(y_test, y_predict_1)
acc_2 = accuracy_score(y_test, y_predict_2)
acc_3 = accuracy_score(y_test, y_predict_3)
acc_4 = accuracy_score(y_test, y_predict_4)
acc_5 = accuracy_score(y_test, y_predict_5)
acc_6 = accuracy_score(y_test, y_predict_6)
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


# LinearSVC_accuracy :  1.0
# Perceptron_accuracy :  1.0
# LogisticRegression_accuracy :  1.0
# KNeighborsClassifier_accuracy :  1.0
# DecisionTreeClassifier_accuracy :  1.0
# RandomForestClassifier_accuracy :  1.0

# LinearSVC_결과 acc :  1.0
# Perceptron_결과 acc :  1.0
# LogisticRegression_결과 acc :  1.0
# KNeighborsClassifier_결과 acc :  1.0
# DecisionTreeClassifier_결과 acc :  1.0
# RandomForestClassifier_결과 acc :  1.0






