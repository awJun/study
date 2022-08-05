from unittest import result         
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston   # sklearn은 학습 예제가 많음

# from sklearn import 전처리
#  from sklearn import utils #y 값을 범주형 값으로 변환
#  lab = 전처리. LabelEncoder () 
# y_transformed = 연구실. fit_transform (y) # 변환된 값
#  보기 인쇄 (y_transformed) 
# [0 1 1 0]

# from sklearn import preprocessing
# from sklearn import utils

# lab = preprocessing.LabelEncoder()
# y_transformed = lab.fit_transform(y)

# #view transformed values
# print(y_transformed)

#1. 데이터
datasets = load_boston()   
x = datasets.data    # 데이터가
y = datasets.target  # y에 들어간다.

print(x)
print(y)

print(x.shape, y.shape) # (506, 13) (506,)  열 13    (506, ) 506개 스칼라, 1개의 백터
                        # intput (506, 13), output 1
print(datasets.feature_names)
 # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 # 'B' 'LSTAT']
 
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model_1 = LinearSVC()
model_2 = Perceptron()
model_3 = LinearRegression()
model_4 = KNeighborsClassifier()
model_5 = DecisionTreeClassifier()
model_6 = RandomForestClassifier()

from sklearn import preprocessing

lab = preprocessing.LabelEncoder()
x_train = lab.fit_transform(x_train)
x_test = lab.fit_transform(x_test)
y_train = lab.fit_transform(y_train)
y_test = lab.fit_transform(y_test)

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





