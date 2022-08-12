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

#1. 데이터
# datasets = load_iris()               # (150, 4)  -->  (150, 2)  /  (150, 1)도 해볼 것
# datasets = load_breast_cancer()      # (569, 30) -->  (569, 1)
# datasets = load_wine()               # (178, 13) -->  (178, 2)
# datasets = fetch_covtype()           # (581012, 54) --> (581012, 6)  7개에서 6개로 압축됨
datasets = load_digits()               # (1797, 64)  -->  (1797, 9)  9개

x = datasets.data
y = datasets.target
print(x.shape)

# 스케일링을한 다음에 데이터를 나누는 경우가 좀 더 좋다고함 통상적으로 그냥 실험해보셈 

lda = LinearDiscriminantAnalysis()   # <-- 디폴트는 -1로 들어간다

lda.fit(x,y)
x = lda.transform(x)
print(x.shape)

lda_EVR = lda.explained_variance_ratio_    
print(lda_EVR)
cumsum = np.cumsum(lda_EVR)
print(cumsum)



