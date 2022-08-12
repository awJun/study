""""
[핵심]
print(datasets.feature_names)로 해당 데이터의 컬럼들의 이름을 확인하고

df = pd.DataFrame(x, columns=[['sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)']])

x = datasets['data']에서 np로 받은 데이터를 괄호안에 컬럼들의 이름들을 넣어서 df로 변환함


[ df에 y를 추가 ]
df['target(Y)'] = y  # df에 y를 넣는데 그 이름을 target(Y)로 하겠다
# print(df)  [150 rows x 5 columns]


[ corr 설명 ]
print(df.corr())    #  .corr() 상관관계를 알려준다.

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()

상관관계에 대한 그래프를 그린 것임.
그래프를 그릴때 df과 np모두 그릴 수 있다 / 두 개의 차이점은 그래프 형태가 조금 달라진다.


[ 상관계수 ]
1. 상관계수 r은 항상 -1과 1 사이에 있다.


"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
print(datasets.feature_names)

x = datasets['data']
y = datasets['target']

df = pd.DataFrame(x, columns=[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])
# print(df)  [150 rows x 4 columns]

df['target(Y)'] = y  # df에 y를 넣는데 그 이름을 target(Y)로 하겠다
# print(df)  [150 rows x 5 columns]

# numpy는 데이터만 들어가잇고 컬럼과 row가 안들어가 있는 상태
# df는 컬럼과 row가 다 들어가 있다

print("================ 상관계수 히트 맵 ==========================================================")
print(df.corr())   # .corr() 상관관계를 알려준다.

import matplotlib.pyplot as plt
import seaborn as sns   # 그래프를 그릴때 추가 양식같은 느낌? 같음
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()





