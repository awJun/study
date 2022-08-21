"""
[핵심]
비지도학습은 분류에서만 사용할 수 있다.

KMeans는 비지도 학습으로 특성을 받아서 그 특성들의 공통점을 찾아서

kmeans = KMeans(n_clusters=3, random_state=1234)

의 n_clusters 부분에서 원하는 라벨의 갯수를 정주면 KMeans가 공통점을 분석해서 라벨을 찾아준다
이때 n_clusters의 갯수는 데이터의 원래 라벨에 맞춰주는 것이 좋다.

이것의 활용으로는
결측치들의 데이터를 모아서 KMeans를 사용해서 결측치의 값의 공통점을 찾아서 결측치를 해결을 할 수도 있다.


"""


from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

datasets = load_iris()

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])   # 컬럼 이름을 datasets의 컬럼 이름으로 하겠다.
# print(df)
#    sepal length (cm) sepal width (cm) petal length (cm) petal width (cm)
# 0                 5.1              3.5               1.4              0.2
# 1                 4.9              3.0               1.4              0.2
# 2                 4.7              3.2               1.3              0.2
# 3                 4.6              3.1               1.5              0.2
# 4                 5.0              3.6               1.4              0.2
# ..                ...              ...               ...              ...
# 145               6.7              3.0               5.2              2.3
# 146               6.3              2.5               5.0              1.9
# 147               6.5              3.0               5.2              2.0
# 148               6.2              3.4               5.4              2.3
# 149               5.9              3.0               5.1              1.8

kmeans = KMeans(n_clusters=3, random_state=1234)   # n_clusters : 라벨의 갯수 웬만하면 원래 라벨갯수와 동일하게 하는 것이 좋다.
kmeans.fit(df)
# print(kmeans.labels_)   # 결과값이 KMeans에서 구한 라벨값으로 나온다.
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2 2 2 1 2 2 2 2
#  2 2 1 1 2 2 2 2 1 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 2 2 2 1 2 2 2 1 2 2 2 1 2
#  2 1]

print(datasets.target)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]


# 두 개가 완전히 동일하진 않지만 얼추 비슷하다는 것을 볼 수 있다.  아래에서 얼마나 비슷한지  accuracy를 사용해서 비교해보겠다.



# 위에껄로 그냥 사용해도 괜찮지만 이렇게 해도 상관없음.
cluster = df['cluster'] = kmeans.labels_
target = df['target'] = datasets.target



#4. 평가, 예측
score = accuracy_score(target, cluster)
print("두 라벨의 정확도 : ", score)   

# 두 라벨의 정확도 :  0.8933333333333333











