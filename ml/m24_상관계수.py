""""
[핵심]
각각의 컬럼들의 연관성을 확인하기 위해서 사용한다.

print(datasets.feature_names)로 해당 데이터의 컬럼들의 이름을 확인하고

### [ 해당 과정에서 사용한 데이터는 iris라이브러리이므로해당 과정을 해준것임 아니면 에러남]####
df = pd.DataFrame(x, columns=[['sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)']])

x = datasets['data']에서 np로 받은 데이터를 괄호안에 컬럼들의 이름들을 넣어서 df로 변환함
################################################################################################

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

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']

df = pd.DataFrame(x, columns=datasets.feature_names)
df['Target(Y)'] = y
print(df)

print('====================== 상관계수 히트 맵 ======================')
print(df.corr()) # 각 칼럼별로 서로 상관관계를 나타냄, 단순리니어모델로 쭉 돌려보고 나온 결과치니까 무조건 신뢰하기 힘듦, 신뢰도 7~80%
# 양의 상관계수 = 비례, 음의 상관계수 = 반비례
'''
                   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target(Y)
sepal length (cm)           1.000000         -0.117570           0.871754          0.817941   0.782561
sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126  -0.426658
petal length (cm)           0.871754         -0.428440           1.000000          0.962865   0.949035
petal width (cm)            0.817941         -0.366126           0.962865          1.000000   0.956547
Target(Y)                   0.782561         -0.426658           0.949035          0.956547   1.000000
'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()





