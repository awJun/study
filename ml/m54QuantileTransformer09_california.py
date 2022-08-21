"""
[핵심]
scaler = [MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer] 
위에  스케일러들 테스트임

PowerTransformer(method='yeo_johnson')   QuantileTransformer(method='BOX_COX')   메소드를 넣으면 에러 발생해서 일단 빼고 돌렸음 ..ㅠ

"""
from sklearn.datasets import fetch_california_housing
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

#1. 데이터
datasets = fetch_california_housing()
x, y = datasets.data, datasets.target
# print(x.shape, y.shape)    # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y,     # 판다스 먹힌다. 
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=1234,
                                                    )

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
    
    
    
# MinMaxScaler의 결과 :  0.8054
# MaxAbsScaler의 결과 :  0.8039
# StandardScaler의 결과 :  0.8047
# RobustScaler의 결과 :  0.8039
# QuantileTransformer의 결과 :  0.806
# PowerTransformer의 결과 :  0.8033


