import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

# print(data)
data = data.transpose()   # 행과 열 위치 변환
# print(data)
data.columns = ["x1", "x2", "x3", "x4"]
# print(data.shape)   # (4, 5)
print(data)

from sklearn.experimental import enable_iterative_imputer     # experimental : 실험적인  / IterativeImputer를 사용하기 위해서 위에서 선언시킴 이거 안하면 에러남 
                                                                # 애는 사용안함 그냥 IterativeImputer를 사용하려고 위에 선언만하는 용도임!
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer  # 비지도학습에서 ~~? 찾아보셈   # 이거 안돌아가서 위에서 실험모델 가져옴
from sklearn.impute import IterativeImputer
imputer = IterativeImputer()   # 결측치 처리를 해준다.  디폴트 mean
# imputer = SimpleImputer(strategy='mean')            # 평균값으로 채우겠다.  
# imputer = SimpleImputer(strategy='median')          # 중위값으로 채우겠다.
# imputer = SimpleImputer(strategy='most_frequent')   # 가장 빈번하게 쓰는 놈을 채우겠다.
# imputer = SimpleImputer(strategy='constant', fill_value=77777)   # 상속하겠다. fill_value의 값으로 / 디폴트는 0으로 들어간다.


imputer.fit(data)
data2 = imputer.transform(data)
print(data2)






