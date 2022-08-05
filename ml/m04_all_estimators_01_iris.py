from logging import warning
import numpy as np
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
# warnings.filterwarnings('ignore')    
                           
#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)  #행(Instances): 150   /   열(Attributes): 4
# print(datasets.feature_names)

x = datasets['data']  # .data와 동일 
y = datasets['target']  
# print(x.shape)   # (150, 4)
# print(y.shape)   # (150,)
# print("y의 라벨값 : ", np.unique(y))  # 해당 데이터의 고유값을 출력해준다.


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
Allalgorithm = all_estimators(type_filter='classifier')
print("algorithm : ", Allalgorithm)
print("모델의 갯수 : ", len(Allalgorithm))   # 모델의 갯수 :  41

for (name, algorithm) in Allalgorithm:
    try:   # 예외처리
        model  = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        # continue    # continue : 계속 진행해
        print(name, '은 안나온 놈!!!')
    
# 3. 컴파일, 훈련
# model.fit(x_train, y_train)


# #4. 평가, 예측
# from sklearn.metrics import accuracy_score
# y_predict = model.predict(x_test)

# acc = accuracy_score(y_test, y_predict)

# print('LinearSVC_accuracy : ', acc)

# results_1 = model.score(x_test, y_test)

# print("LinearSVC_결과 acc : ", results_1)   # 회기는 r2 / 분류는 acc로 결과가 나온다.







# # 딥러닝과 머신러닝 차이
# # 딥러닝은 레이어를 길게 뺀거
# # 머신러닝은 간결해서 속도가 빠르다.


# # 원핫 할 필요없음 모델구성에서 알아서 받아짐
# # 컴파일 없음 훈련도 x y만 하면 된다. fit에 컴파일이 아랑서 포함되어 있다 그러므로 컴파일이 없음
# # 훈련에서 튜닝하고 평가랗때 이벨류에이트없고 스코어를 사용한다.

