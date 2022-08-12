"""
[핵심]
PCA는 차원을 압축시키는데 이때 열을 압축시킨다.

pca = PCA(n_components=13)  # 2개로 압축하겠다. 
x = pca.fit_transform(x)
print(x.shape)  

PCA는 대표적인 비지도 학습중 하나임 (비지도학습 y가 없다?)

[ PCA 사용하면? ]
성능이 더 좋아지거나 비슷하거나 살짝 더 안좋아진다.
차원을 압축해서 열의 갯수가 줄어들음으로써 훈련시간이 감소된다. ! 즉 차웝이 줄고 성능이 같은 경우는
좋은 축에 속한다!

[ 주의점 ]
압축을 할 때 범위는 데이터셋에 열의 갯수보다 높은 숫자를 넣으면 에러가 발생한다. 이것은 압축이 아니라 증폭
개념이기 때문이다.

"""

# 맹그러 테스터 

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

for i in range(x.shape[1]):     # range를 사용하면 0부터 29까지이므로 +1을 해준다
    pca = PCA(n_components=i+1)   # 0부터 들어가므로 +1
    x2 = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, random_state=123, shuffle=True)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(i+1, '의 결과: ', results)
    




# 모든 칼럼
# 결과:  0.7573250241545894

# 14
# 결과:  0.8847468760441028

# 4
# 결과:  0.9011631807550952








