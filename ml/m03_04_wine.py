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
 
from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.9, shuffle=True, random_state=86)


# 2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # !논리회귀(분류임)!
from sklearn.neighbors import KNeighborsClassifier # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 봄

model = LinearSVC()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('LinearSVC acc 결과: ', result)
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')

model = SVC()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('SVC acc 결과: ', result)
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')

model = Perceptron()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('Perceptron acc 결과: ', result)
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')

model = LogisticRegression()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('LogisticRegression acc 결과: ', result)
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')

model = KNeighborsClassifier()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('KNeighborsClassifier acc 결과: ', result)
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('DecisionTreeClassifier acc 결과: ', result)
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')

model = RandomForestClassifier()
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print('RandomForestClassifier acc 결과: ', result)
y_predict = model.predict(x_test)
print('ypred: ', y_predict, '\n')


# LinearSVC acc 결과:  0.9444444444444444
# ypred:  [1 0 0 2 0 1 1 2 1 1 1 2 0 1 2 1 2 1]

# SVC acc 결과:  0.6666666666666666
# ypred:  [1 0 0 2 0 1 2 2 1 1 1 1 0 2 1 2 0 1]

# Perceptron acc 결과:  0.5555555555555556
# ypred:  [1 0 0 0 0 1 0 1 1 1 1 1 0 0 1 1 0 1]

# LogisticRegression acc 결과:  1.0
# ypred:  [1 0 0 2 0 1 1 2 1 1 1 2 0 1 2 0 2 1]

# KNeighborsClassifier acc 결과:  0.7222222222222222
# ypred:  [1 0 0 1 0 1 2 1 1 1 1 2 0 2 2 0 2 2]

# DecisionTreeClassifier acc 결과:  0.9444444444444444
# ypred:  [1 0 0 2 0 1 1 2 1 1 1 0 0 1 2 0 2 1]

# RandomForestClassifier acc 결과:  1.0
# ypred:  [1 0 0 2 0 1 1 2 1 1 1 2 0 1 2 0 2 1]