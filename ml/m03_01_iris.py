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
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression는 분류임
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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






# 딥러닝과 머신러닝 차이
# 딥러닝은 레이어를 길게 뺀거
# 머신러닝은 간결해서 속도가 빠르다.


# 원핫 할 필요없음 모델구성에서 알아서 받아짐
# 컴파일 없음 훈련도 x y만 하면 된다. fit에 컴파일이 아랑서 포함되어 있다 그러므로 컴파일이 없음
# 훈련에서 튜닝하고 평가랗때 이벨류에이트없고 스코어를 사용한다.

