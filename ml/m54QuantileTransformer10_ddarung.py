"""
[핵심]
scaler = [MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer] 
위에  스케일러들 테스트임

PowerTransformer(method='yeo_johnson')   QuantileTransformer(method='BOX_COX')   메소드를 넣으면 에러 발생해서 일단 빼고 돌렸음 ..ㅠ

"""


# Dacon 따릉이 문제풀이
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold,\
    HalvingRandomSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함


# 1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

### 결측치 처리(일단 제거로 처리) ###
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
# train_set = train_set.dropna() # nan 값(결측치) 열 없앰
train_set = train_set.fillna(0) # 결측치 0으로 채움
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

###[ 핵심 ]################################################################################################################

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor

scaler = [MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer]


for scalers in scaler:
    scalers = scalers()
    scaler_name = str(scalers).strip('()')  # .strip('()')참고 https://ai-youngjun.tistory.com/68
    x_train = scalers.fit_transform(x_train)
    x_test = scalers.transform(x_test)
    
    model = RandomForestRegressor()
    model.fit(x_train, y_train)
    
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    print(scaler_name + "의 결과 : ", round(results, 4))
    
    
    
# MinMaxScaler의 결과 :  0.7849
# MaxAbsScaler의 결과 :  0.7804
# StandardScaler의 결과 :  0.7864
# RobustScaler의 결과 :  0.7869
# QuantileTransformer의 결과 :  0.7838
# PowerTransformer의 결과 :  0.7941

