"""
all_estimators
 - 해당 데이터셋에서 사용할 수 있는 모든 알고리즘 모델들을 가져와준다.


사용법
Allalgorithm = all_estimators(type_filter='classifier')  # 분류 모델이므로 classifier
print("algorithm : ", Allalgorithm)
print("모델의 갯수 : ", len(Allalgorithm))   # 모델의 갯수 :  41  

 # 모델의 갯수 = 해당 데이터셋에서 사용할 수 있는 모델의 갯수를 뜻한다.


버전이 달라서 안돌아가는 경우를 생각해서 예외처리 및 출력작업
for (name, algorithm) in Allalgorithm:    # Allalgorithm는 딕셔너리이므로 키와 벨류로 받아서 반복시킴
    try:   # 예외처리
        model  = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        print(name, '은 안나온 놈!!!')
설명: try(시도)해라 아래 내용들을 내용중 안되는 것은 except(제외하고) 진행해라

import warnings
warnings.filterwarnings('ignore')    
 - except를 사용해서 예외로 처리할 땐 무조건 warning을 불러와야한다. 아니면 에러발생함


"""


import numpy as np
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')    
                           
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
# print("algorithm : ", Allalgorithm)
# print("모델의 갯수 : ", len(Allalgorithm))   # 모델의 갯수 :  41

for (name, algorithm) in Allalgorithm:
    try:   # 예외처리
        model  = algorithm()
        model.fit(x_train, y_train)      # 훈련 및 컴파일
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)  # 성능확인
    except:
        # continue    # continue : 계속 진행해
        print(name, '은 안나온 놈!!!')    # 안돌아가는 모델

  
  
