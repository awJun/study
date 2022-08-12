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







