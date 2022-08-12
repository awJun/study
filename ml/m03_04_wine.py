"""
[핵심]
각 모델 성능 테스트  (해당 모델들은 분류 모델들이다.)
model_1 = LinearSVC()
model_2 = Perceptron()
model_3 = LogisticRegression()
model_4 = KNeighborsClassifier()
model_5 = DecisionTreeClassifier()
model_6 = RandomForestClassifier()

[중요]
LogisticRegression
- 이것은 Regression들어가지만 분류 모델이다.
- LinearRegression 이친구가 회기 모델이다.
이 친구 빼고는 나머지는 다 Regression이 들어가면 회기 모델로 생각하면 된다.

Classifier가 들어가면 분류 모델로 생각하면 된다.

"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.datasets import load_wine
import tensorflow as tf
from sklearn.svm import LinearSVC, LinearSVR


#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13), (178,)
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

# # scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 0.0
# print(np.max(x_train))  # 1.0

# print(np.min(x_test))  # 1.0
# print(np.max(x_test))  # 1.0

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

# LinearSVC_accuracy :  0.9074074074074074
# Perceptron_accuracy :  0.7407407407407407
# LogisticRegression_accuracy :  0.7407407407407407
# KNeighborsClassifier_accuracy :  0.7407407407407407
# DecisionTreeClassifier_accuracy :  0.7407407407407407
# RandomForestClassifier_accuracy :  0.7407407407407407

# LinearSVC_결과 acc :  0.9074074074074074
# Perceptron_결과 acc :  0.7407407407407407
# LogisticRegression_결과 acc :  0.7407407407407407
# KNeighborsClassifier_결과 acc :  0.7407407407407407
# DecisionTreeClassifier_결과 acc :  0.7407407407407407
# RandomForestClassifier_결과 acc :  0.7407407407407407