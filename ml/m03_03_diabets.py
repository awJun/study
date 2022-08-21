"""
[핵심]
각 모델 성능 테스트  (해당 모델들은 회기 모델들이다.)
model_1 = LinearSVR()
model_2 = Perceptron()
model_3 = LinearRegression()
model_4 = KNeighborsRegressor()
model_5 = DecisionTreeRegressor()
model_6 = RandomForestRegressor()

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
 
from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.9, shuffle=True, random_state=86)


# 2. 모델구성
from sklearn.svm import LinearSVR, SVR, LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

models = [LinearSVR, SVR, Perceptron, LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]

for model in models:
    model = model()
    model_name = str(model).strip('()')
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(model_name, '결과: ', result)
    
# LinearSVR 결과:  -0.21026793010486222
# SVR 결과:  0.2668794571758186
# Perceptron 결과:  0.0
# LinearRegression 결과:  0.6557534150889773
# KNeighborsRegressor 결과:  0.5704639112420011
# DecisionTreeRegressor 결과:  -0.054909125498723066
# RandomForestRegressor 결과:  0.6329634639445246