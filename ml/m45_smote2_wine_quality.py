# 실습!! 시작!!

# smote로 증폭
# 안나누고 smote를 해봐라 ~~



import pandas as pd
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

path = "D:/study_data/_data/"

datasets = pd.read_csv(path+'winequality-white.csv',index_col=None, header=0, sep=';')

# print(datasets.shape)  # (4898, 12)
# print(datasets.describe())   # pandas에서만 사용가능
# print(datasets.info())   # 결측치 확인

import numpy as np  # 조금 더 빠르다 ..? 뭐지..
#--[ 둘 다 같은 방법임]--------
datasets2 = datasets.values
# datasets = datasets.to_numpy()     # numpy는 인덱스와 컬럼이 없다.
#---------------------------------
# x = datasets.drop(['quality'], axis=1)
# y = datasets['quality']
x = datasets2[:, :11]   # 11째 까지
y = datasets2[:, 11]    # 11 위치에 잇는 거

x_new = x[:-25]
y_new = y[:-25]
# print(pd.Series(y_new).value_counts())
# 6.0    2184
# 5.0    1451
# 7.0     876
# 8.0     175
# 4.0     162
# 3.0      20
# 9.0       5


print(x.shape, y.shape)  # (4898, 11) (4898,)

print(np.unique(y, return_counts=True))
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# 분류에서는 다중이든 이진이든 확인할것

print(datasets["quality"].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=123,
                                                    )

# print(pd.Series(y_train).value_counts())
# 6.0    1732
# 5.0    1183
# 7.0     714
# 8.0     136
# 4.0     133
# 3.0      18
# 9.0       2

from imblearn.over_sampling import SMOTE 
smote = SMOTE(random_state=123, k_neighbors=1)     # SMOTE는 증폭!  /  훈련 데이터만 증폭할 수 있다.
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts()) 


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)



#4. 평가, 예측
y_predict = model.predict(x_test)

score = model.score(x_test, y_test)
print("model.score : ", score)

from sklearn.metrics import accuracy_score, f1_score
print('acc_score : ', accuracy_score(y_test, y_predict))
print("f1_score(macro) : ", f1_score(y_test, y_predict, average='macro'))
# print("f1_score(micro) : ", f1_score(y_test, y_predict, average='micro'))



# model.score :  0.6612244897959184
# acc_score :  0.6612244897959184
# f1_score(macro) :  0.3937015018016852





















































