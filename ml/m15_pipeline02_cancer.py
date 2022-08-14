import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1234)

# 2. 모델
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

model = make_pipeline(MinMaxScaler(), RandomForestClassifier()) # 굳이 변수명으로 정의하지 않아도 바로 갖다 쓸 수 있음

# 3. 훈련
model.fit(x_train, y_train) # pipeline의 fit에는 알아서 fit_transform이 들어가 있음

# 4. 평가, 예측
result = model.score(x_test, y_test) # pipeline의 score에는 알아서 transform이 들어가 있음

print('model.score: ', result)

# model.score:  0.9210526315789473

# 스케일링 안했을 때
# LinearSVC 결과:  0.8508771929824561
# SVC 결과:  0.8947368421052632
# Perceptron 결과:  0.8947368421052632
# LogisticRegression 결과:  0.956140350877193
# KNeighborsClassifier 결과:  0.9210526315789473
# DecisionTreeClassifier 결과:  0.9210526315789473
# RandomForestClassifier 결과:  0.956140350877193