# 라벨 1 357
# 라벨 0 212

# 라벨 0을 112개 삭제해서 재구성

# smote 넣어서 맹그러
# 넣은거 안넣은거 비교 


# smote를 넣어서 넣은 것과 안넣은 것을 비교

# 1. 데이터
import numpy as np
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0
#  1 0 1 0 0 1 1 1 0 0 1 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1
#  1 1 1 1 1 1 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 0 1 0 0 1 1 0 1 1 0 1 1 1 1 0 1
#  1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 1 1 0 1 1 0 0 0 1 0
#  1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 1 0 0 0 0 1 1 0 0 1 1
#  1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 1 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1
#  1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 1 1
#  1 1 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0
#  0 1 0 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1
#  1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 1 1
#  0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1
#  1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 0 1 0 1 0 1 1 1 1 1 0 1 1 0 1 0 1 0 0
#  1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 0 0 0 0 0 0 1]

# print(np.unique(y, return_counts=True))  
# (array([0, 1]), array([212, 357], dtype=int64))

newlist = []
for i in y:  # y를 다 넣음   y 범위 3 ~ 9
    if i == 0:
        newlist += [0]    #  
        if len(newlist) < 113:
            break
    else:
        newlist += [1]    # 

print(np.unique(newlist, return_counts=True))  

# # print(x.shape, y.shape)   # (569, 30) (569,)


# # print(np.unique(y, return_counts=True))   
# # (array([0, 1]), array([212, 357], dtype=int64))


# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# x_train, x_test, y_train, y_test = train_test_split(x, y,
#                                                     train_size=0.8,
#                                                     shuffle=True,
#                                                     random_state=123,
#                                                     )

# import pandas as pd
# # print(pd.Series(y_train).value_counts())
# # 1    284
# # 0    171

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()

# from imblearn.over_sampling import SMOTE 
# smote = SMOTE(random_state=123)     # SMOTE는 증폭!  /  훈련 데이터만 증폭할 수 있다.
# x_train, y_train = smote.fit_resample(x_train, y_train)

# print(pd.Series(y_train).value_counts())    # 증폭하면 훈련할 때 시간이 2배로 걸린다.
# # 0    284
# # 1    284


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

# ###[ 증폭 전 ]#################################
# # model.score :  0.9912280701754386
# # acc_score :  0.9912280701754386
# # f1_score(macro) :  0.9904257999496096


# ###[ 증폭 후 ]#################################
# # model.score :  0.9912280701754386
# # acc_score :  0.9912280701754386
# # f1_score(macro) :  0.9904257999496096














