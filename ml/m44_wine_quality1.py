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


# model.score :  0.6928571428571428
# acc_score :  0.6928571428571428
# f1_score(macro) :  0.39742311190946145



