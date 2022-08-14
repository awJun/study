"""

LDA = from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 로 선언 후 사용

[주의점]
n_components 사용할 때 y의 라벨보다 큰수를 넣으면 안된다
라벨 갯수보다 많으면 ValueError: n_components cannot be larger than min(n_features, n_classes - 1). 발생

explained_variance_ratio_을 사용하면 분포도가 ??

pca = PCA(n_components=3)
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_   # 여기서 중요도를 뽑아줌
print(pca_EVR) 
cumsum = np.cumsum(pca_EVR)               # explained_variance_ratio_에서 뽑은 값을 더해서 왼쪽부터 순서대로 출력해줌 즉! 누적합
print(cumsum)

[값 비교]
[0.92461872 0.05306648 0.01710261]
[0.92461872 0.97768521 0.99478782]   

92461872 + 05306648 = 97768521 <-- 두번째부분
97768521 + 01710261 = 99478782 <-- 세번째 부분

[PCA와 LDA차이]
 - PCA는 비지도 LDA는 지도학습을 수행한다.
 - PCA는 x값 축소 LDA는 y값 축소를 하는 거 같음

"""


# import xgboost as xg
# print(xg.__version__) # 1.6.1

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape)
# le = LabelEncoder()  // stratify=y 쓰면 라벨인코더 안해도 됨
# y = le.fit_transform(y)

pca = PCA(n_components=x.shape[1])
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
cumsum = np.cumsum(pca_EVR)
print(cumsum)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, shuffle=True, stratify=y)
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([169472, 226640,  28603,   2198,   7594,  13894,  16408],
#       dtype=int64))
lda = LinearDiscriminantAnalysis(n_components=1)
lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

# 2. 모델
from xgboost import XGBClassifier
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1)

# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과: ', results)
print('걸린 시간: ', end-start)

# xgboost - gpu
# 결과:  0.8695988915948814
# 걸린 시간:  5.994714736938477

# xgboost - gpu / PCA n_component - 10
# 결과:  0.8406065247885166
# 걸린 시간:  4.38213324546814

# xgboost - gpu / PCA n_component - 20
# 결과:  0.8857946868841596
# 걸린 시간:  4.646213531494141

# PCA는 y값을 건들지 않고 x값만 축소하지만
# LDA는 y값을 x값 축소할 때 같이 연산에 포함한다

# LDA n_components = 1
# 결과:  0.9
# 걸린 시간:  0.9540712833404541

# LDA n_components = 2
# 결과:  0.9
# 걸린 시간:  0.4767777919769287
