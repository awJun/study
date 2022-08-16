# fetch_covtype는 컬럼이 적어서 생각보다 오래 안걸릴거야 ~~ ㅋ

# 실습
# 증폭한 후 저장

# 증폭 시간 체크하기  / 증폭시간을 잴것이므로 fit_resample에서 시간 재면됨 



from sklearn.datasets import load_breast_cancer, fetch_covtype


# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

import pandas as pd
# print(pd.Series(y).value_counts())
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

import numpy as np
# print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]) 
#  , array([211840, 283301,  35754,   2747,   9493,  17367,  20510],

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234, train_size=0.8)

# print(pd.Series(y_train).value_counts())
# 2    226613
# 1    169671
# 3     28468
# 7     16380
# 6     13852
# 5      7618
# 4      2207

import time
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123)     # SMOTE는 증폭!  /  훈련 데이터만 증폭할 수 있다.

start = time.time()
x_train, y_train = smote.fit_resample(x_train, y_train)
end = time.time() -start
print(pd.Series(y_train).value_counts())    # 증폭하면 훈련할 때 시간이 2배로 걸린다.
# 2    226613
# 7    226613
# 1    226613
# 3    226613
# 6    226613
# 5    226613
# 4    226613

print(end)

# import pickle
# path = 'd:/study_data/_save/_xg/m45_smote7/'
# pickle.dump(x_train, open(path + 'x_train_save.dat', 'wb'))      # dump로 저장함
# pickle.dump(y_train, open(path + 'y_train_save.dat', 'wb'))      # dump로 저장함
# pickle.dump(x_test, open(path + 'x_test_save.dat', 'wb'))      # dump로 저장함
# pickle.dump(y_test, open(path + 'y_test_save.dat', 'wb'))      # dump로 저장함



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
