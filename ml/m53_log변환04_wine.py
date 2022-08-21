from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

datasets = load_wine()
x, y = datasets.data, datasets.target
# print(x.shape, y.shape)    # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=1234,
                                                    )


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
from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
results = r2_score(y_test, y_predict)
print("로그변환 결과 : ", (results, 4))




# 기냥 결과 :  (0.8531400966183575, 4)
# 로그변환 결과 :  (0.7797101449275362, 4)

# 결론 성능이 더 안좋아짐