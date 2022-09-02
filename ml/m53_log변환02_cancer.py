"""
[핵심]
데이터 전처리의 로그변환 _스케일러보다 좋은 경우가 많다.
성능향상에도 좋고 그래프 그리기도 좋다.

log100 = 2
log10 = 1
log1 = 0

너무 데이터가 커서 딥러닝이 불가능한 경우 유용하다.

큰 숫자를 줄여주는 역할임 즉! 훈련할 때 유용함

단! log를 사용할 때
pridict 할 때 안에 넣는 값은 log를 하고 난 이후에 넣어야한다.



"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, accuracy_score


#1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target
# print(x.shape, y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x, y,     # 판다스 먹힌다. 
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=1234,
                                                    )

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler,\
                                  RobustScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


#2. 모델
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
results = r2_score(y_test, y_predict)
print("기냥 결과 : ", (results, 4))


##################### 로그 변환 #############################################

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
# print(df.columns)

###[ 이상치 확인 후 이상있는 컬럼 선정 후 log1p 적용 ]


# import matplotlib.pyplot as plt
# df.plot.box()
# plt.title('boston')
# plt.xlabel("Feature")
# plt.ylabel("데이터 값")
# plt.show()


# print(df['worst perimeter'].head())
# df['sepal width (cm)'] = np.log1p(df['sepal width (cm)'])
# print(df['sepal width (cm)'].head())


df['mean area'] = np.log1p(df['mean area'])       # 값이 1000 대
df['area error'] = np.log1p(df['area error'])     # 값이 153 ~ 27대
df['worst perimeter'] = np.log1p(df['worst perimeter'])  # 100대
df['worst area'] = np.log1p(df['worst area'])     # 1000대 


x_train, x_test, y_train, y_test = train_test_split(df, y,               # 판다스 먹힌다. 
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=1234,
                                                    )

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)

#2. 모델
model = LogisticRegression()
# model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
results = r2_score(y_test, y_predict)
print("로그변환 결과 : ", (results, 4))




# 기냥 결과 :  (0.8531400966183575, 4)
# 로그변환 결과 :  (0.7797101449275362, 4)

