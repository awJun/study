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

from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

datasets = load_wine()

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names]) 

# print(np.unique(datasets['target'], return_counts=True))   #라벨 갯수 확인
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))


kmeans = KMeans(n_clusters=3, random_state=1234)
kmeans.fit(df)

cluster = df['cluster'] = kmeans.labels_
target = df['target'] = datasets.target

#4. 평가, 예측
score = accuracy_score(target, cluster)
print("두 라벨의 정확도 : ", score)   

# 두 라벨의 정확도 :  0.702247191011236
