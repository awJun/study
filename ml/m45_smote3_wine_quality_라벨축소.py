"""

# y 라벨을 줄일거임 이거는 업무에서만 사용하셈 회사에서는 오더한테 허락받아야하고 캐글에서는 사용불가!##

이걸로 안에 어떤 값이 들었는지 확인하고
# print(np.unique(y, return_counts=True))
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))

아래에서 조건문에서 걸어서 컬럼을 줄임

newlist = []
for i in y:  # y를 다 넣음   y 범위 3 ~ 9
    if i <=5:
        newlist += [0]    # 
    elif i==6:
        newlist += [1]    # 
    else:
        newlist += [2]    # 

를 사용해서 y컬럼을 축소


아래에 이걸 사용한 이유는
x_new = x[:-25]
y_new = y[:-25]
데이터를 줄이고 증폭되는 것을 보여줄려고 그냥 해본거임 신경 ㄴㄴ

증폭할때 걸린시간을 알고싶으면 
x_train, y_train = smote.fit_resample(x_train, y_train)
여기 부분에다가 time.time()을 사용하면 된다.

"""


# 실습 시작 !! 라벨 3개짜리로...

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
x = datasets2[:, :11]   # 컬럼 0부터 11째 까지
y = datasets2[:, 11]    # 컬럼 11 위치에 잇는 거

# print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5


x_new = x[:-25]    # x행의 25개의 데이터를 제거
y_new = y[:-25]    # y행의 25개의 데이터를 제거
# print(pd.Series(y_new).value_counts())
# 6.0    2184
# 5.0    1451
# 7.0     876
# 8.0     175
# 4.0     162
# 3.0      20
# 9.0       5


# print(x.shape, y.shape)  # (4898, 11) (4898,)

# print(np.unique(y, return_counts=True))
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# 분류에서는 다중이든 이진이든 확인할것


newlist = []
for i in y:  # y를 다 넣음   y 범위 3 ~ 9
    if i <=5:
        newlist += [0]    # 
    elif i==6:
        newlist += [1]    # 
    else:
        newlist += [2]    # 
        
print(np.unique(newlist, return_counts=True))    # (array([0, 1, 2]), array([1640, 2198, 1060], dtype=int64))





print(datasets["quality"].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=123,
                                                    stratify=y
                                                    )

# print(pd.Series(y_train).value_counts())
# 6.0    1758
# 5.0    1166
# 7.0     704
# 8.0     140
# 4.0     130
# 3.0      16
# 9.0       4

from imblearn.over_sampling import SMOTE 
smote = SMOTE(random_state=123, k_neighbors=3)     # SMOTE는 증폭!  /  훈련 데이터만 증폭할 수 있다.
x_train, y_train = smote.fit_resample(x_train, y_train)

# print(pd.Series(y_train).value_counts()) 
# 4.0    1758
# 5.0    1758
# 6.0    1758
# 7.0    1758
# 8.0    1758
# 3.0    1758
# 9.0    1758

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


##[ 축소전 ]##############################
# model.score :  0.6928571428571428
# acc_score :  0.6928571428571428
# f1_score(macro) :  0.39742311190946145


##[ 축소후 ]##############################
# model.score :  0.689795918367347
# acc_score :  0.689795918367347
# f1_score(macro) :  0.4352315078124421








