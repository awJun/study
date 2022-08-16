import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE     # pip install imblearn 이거 해야지 인식됨
import sklearn as sk
# print("사이킷런 : ", sk.__version__)    # 사이킷런 :  1.1.2

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape)   # (178, 13) (178,)    
# print(type(x))            # <class 'numpy.ndarray'>
# print(np.unique(y, return_counts=True))   # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# print(pd.Series(y).value_counts())    # 컬럼이 한 개이므로 DataFraim이 아닌 Series로 해야한다.

# print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x_new = x[:-25]
y_new = y[:-25]
# print(pd.Series(y_new).value_counts())
# 1    71
# 0    59
# 2    18

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, shuffle=True, train_size=0.75, random_state=123,
                                                    stratify=y_new)

print(pd.Series(y_train).value_counts())
# 1    53
# 0    44
# 2    14

#2. 모델
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

# #[ 기본 결과 ]##########################
# model.score :  0.9777777777777777
# f1_score(micro) :  0.9777777777777777


# # [ 데이터 축소후 ]####################### (2번 라벨을 30개 줄임)
# model.score :  0.972972972972973
# f1_score(micro) :  0.972972972972973


print("======================== SMOTE 적용 ===========================")
smote = SMOTE(random_state=123)     # SMOTE는 증폭!  /  훈련 데이터만 증폭할 수 있다.
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts())    # 증폭하면 훈련할 때 시간이 2배로 걸린다.
# 0    53
# 1    53
# 2    53

# #2. 모델  /  #3. 훈련
# model = RandomForestClassifier()
# model.fit(x_train, y_train)

# #4. 평가, 예측
# y_predict = model.predict(x_test)

# score = model.score(x_test, y_test)
# print("model.score : ", score)

# from sklearn.metrics import accuracy_score, f1_score
# print('acc_score : ', accuracy_score(y_test, y_predict))
# print("f1_score(macro) : ", f1_score(y_test, y_predict, average='macro'))
# # print("f1_score(micro) : ", f1_score(y_test, y_predict, average='micro'))




# model.score :  0.9428571428571428
# acc_score :  0.9428571428571428
# f1_score(macro) :  0.8596176821983273
# ======================== SMOTE 적용 ===========================
# model.score :  0.9428571428571428
# acc_score :  0.9428571428571428
# f1_score(macro) :  0.8596176821983273



