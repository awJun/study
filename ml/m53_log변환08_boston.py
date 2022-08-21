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

from sklearn.datasets import load_boston
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
datasets = load_boston()
x, y = datasets.data, datasets.target
# print(x.shape, y.shape)    # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y,     # 판다스 먹힌다. 
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=1234,
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


#2. 모델
model = LinearRegression()
# model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
results = r2_score(y_test, y_predict)
print("기냥 결과 : ", (results, 4))
# 기냥 결과 :  0.764718453230043    ->   RF기냥 결과 :  0.8573867689813848


##################### 로그 변환 #############################################

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)

# import matplotlib.pyplot as plt
# df.plot.box()
# plt.title('boston')
# plt.xlabel("Feature")
# plt.ylabel("데이터 값")
# plt.show()


# print(df['B'].head())
df['B'] = np.log1p(df['B'])
# print(df['B'].head())
#         B
# 0  396.90
# 1  396.90
# 2  392.83
# 3  394.63
# 4  396.90
#           B
# 0  5.986201
# 1  5.986201
# 2  5.975919
# 3  5.980479
# 4  5.986201

# df['CRIM'] = np.log1p(df['CRIM'])
# df['ZN'] = np.log1p(df['ZN'])
# df['TAX'] = np.log1p(df['TAX'])

x_train, x_test, y_train, y_test = train_test_split(df, y,               # 판다스 먹힌다. 
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=1234,
                                                    )

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)

#2. 모델
model = LinearRegression()
# model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
results = r2_score(y_test, y_predict)
print("로그변환 결과 : ", (results, 4))



# df['B'] = np.log1p(df['B'])          로그변환 결과 :  (0.7710827448613001, 4)

# df['CRIM'] = np.log1p(df['CRIM'])    로그변환 결과 :  (0.759582163653457, 4)

# df['ZN'] = np.log1p(df['ZN'])        로그변환 결과 :  (0.7733890810577744, 4)

# df['TAX'] = np.log1p(df['TAX'])      로그변환 결과 :  (0.7669259619292876, 4)

# 4개 다
# 로그변환 결과 :  (0.7717060482363763, 4)
