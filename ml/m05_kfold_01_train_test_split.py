import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict, train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
                     
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

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression는 분류임
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = SVC()


#3.4 훈련,  컴파일, 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("ACC : ", scores, '\n cross_val_score : ', round(np.mean(scores), 4))  # round를 사용해서 소수 4번째까지 출력해라 라고함

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC :', acc)


# ACC :  [0.96666667 0.96666667 1.         0.93333333 0.96666667] 
#  cross_val_score :  0.9667









# 딥러닝과 머신러닝 차이
# 딥러닝은 레이어를 길게 뺀거
# 머신러닝은 간결해서 속도가 빠르다.


# 원핫 할 필요없음 모델구성에서 알아서 받아짐
# 컴파일 없음 훈련도 x y만 하면 된다. fit에 컴파일이 아랑서 포함되어 있다 그러므로 컴파일이 없음
# 훈련에서 튜닝하고 평가랗때 이벨류에이트없고 스코어를 사용한다.

