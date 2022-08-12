#애는  gpu사용해라 ~
"""

LDA = from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 로 선언 후 사용

[주의점]
n_components 사용할 때 y의 라벨보다 큰수를 넣으면 안된다
라벨 갯수보다 많으면 ValueError: n_components cannot be larger than min(n_features, n_classes - 1). 발생
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import load_wine, fetch_covtype
from sklearn.datasets  import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

import xgboost as xg
# print('xg버전 : ', xg.__version__)  xg버전 :  1.6.1


# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(np.unique(y))  [0 1]  해당 데이터는 2진 분류여서 유니크 값이 2개이므로  n_components를 무조건 1로 해야한다. 2로하면 하나 마나이기 때문이다.

# print(x.shape) # (581012, 54)
le = LabelEncoder()
y = le.fit_transform(y)

# pca = PCA(n_components=20)
# x = pca.fit_transform(x)
# pca_EVR = pca.explained_variance_ratio_
# cumsum = np.cumsum(pca_EVR)
# print(cumsum)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, shuffle=True, stratify=y)
print(np.unique(y_train, return_counts=True))    # (array([0, 1, 2], dtype=int64), array([47, 57, 38], dtype=int64))

lda = LinearDiscriminantAnalysis(n_components=2) # y라벨 개수보다 작아야만 한다   / 라벨 갯수보다 많으면 ValueError: n_components cannot be larger than min(n_features, n_classes - 1). 발생
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

# LinearDiscriminantAnalysis - 1   # 다중분류이므로 2이하로 축소  3로하면 기존과 똑같아서 하나마나이기 때문이다.
# 결과:  0.9722222222222222
# 걸린 시간:  0.5412087440490723



# PCA는 y값을 건들지 않고 x값만 축소하지만
# LDA는 y값을 x값 축소할 때 같이 연산에 포함한다   

# 즉! PCA x값 축소 LDA는 y값 축소를 하는 거 같음


