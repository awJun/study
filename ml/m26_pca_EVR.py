import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA  # 주성분 분석, 차원축소(압축) : 열

import sklearn as sk
# print(sk.__version__)   0.24.2

import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape)  

pca = PCA(n_components=30)  # 개로 압축하겠다. 
x = pca.fit_transform(x)
# print(x.shape)  

pca_EVR = pca.explained_variance_ratio_  # 변환한 값의 중요도 / 즉 손실안된 데이터 정도
# print(pca_EVR)
# print(sum(pca_EVR))  # 0.9999999999999998

cumsum = np.cumsum(pca_EVR)   # 누적 합    # 그래프를 보면 오름차순으로 성능이 좋아지는 걸 볼 수 있다.
print(cumsum)                 # 주어진 축을 따라 요소의 누적 합계를 반환합니다
                              # 누적이란 여러 개의 데이터를 합산하여 하나로 만든 것을 말한다

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()




