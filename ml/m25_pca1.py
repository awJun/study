"""
[핵심]
차원축소 = 열 압축!
PCA는 차원을 압축시키는데 이때 열을 압축시킨다.

pca = PCA(n_components=13)  # 13개로 압축하겠다. 
x = pca.fit_transform(x)
print(x.shape)  

PCA는 대표적인 비지도 학습중 하나임 (비지도학습 y가 없다) 즉! 정답없이 모델 스스로 훈련을 하는 것!

[ PCA 사용하면? ]
성능이 더 좋아지거나 비슷하거나 살짝 더 안좋아진다.
차원을 압축해서 열의 갯수가 줄어들음으로써 훈련시간이 감소된다. ! 즉 차웝이 줄고 성능이 같은 경우는
좋은 축에 속한다!

[ 주의점 ]
압축을 할 때 범위는 데이터셋에 열의 갯수보다 높은 숫자를 넣으면 에러가 발생한다. 이것은 압축이 아니라 증폭
개념이기 때문이다.

"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA  # 차원축소 즉! 압축
import warnings
warnings.filterwarnings(action='ignore')

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

pca = PCA(n_components=12) # 주성분 분석, 차원 축소 // n_components 개수만큼으로 x의 열을 줄임
# y 값이 없는 대표적 비지도 학습의 하나
x = pca.fit_transform(x) 
print(x.shape) # (506, 2)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, shuffle=True)

# 2. 모델
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과: ', results)

# 모든 칼럼
# 결과:  0.9187290554452663

# PCA 12
# 결과:  0.8563581772863746