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


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

# 1. 데이터
datasets = load_iris()
print(datasets.DESCR)
'''
- class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
y값이 3개
이 3개 꽃 중 하나가 나와야 함
3중 분류
'''
print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
print(x, '\n', y)
print(x.shape) # (150, 4)
print(y.shape) # (150,)
print('y의 라벨값: ', np.unique(y))


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
         
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
             
#2. 모델구성
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgorithms: ', allAlgorithms)
print('모델의 개수: ', len(allAlgorithms)) # 41

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        ypred = model.predict(x_test)
        acc = accuracy_score(y_test, ypred)
        print(name, '의 정답률: ', acc)
    except:
        # continue # 또는 pass
        print(name, '은 안나온 놈')

# 버전에 따라 안돌아가는 모델들이 있음

#AdaBoostClassifier 의 정답률:  1.0
# BaggingClassifier 의 정답률:  1.0
# BernoulliNB 의 정답률:  0.36666666666666664
# CalibratedClassifierCV 의 정답률:  1.0
# CategoricalNB 은 안나온 놈
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률:  0.7
# DecisionTreeClassifier 의 정답률:  1.0
# DummyClassifier 의 정답률:  0.26666666666666666
# ExtraTreeClassifier 의 정답률:  1.0
# ExtraTreesClassifier 의 정답률:  1.0
# GaussianNB 의 정답률:  1.0
# GaussianProcessClassifier 의 정답률:  1.0
# GradientBoostingClassifier 의 정답률:  1.0
# HistGradientBoostingClassifier 의 정답률:  1.0
# KNeighborsClassifier 의 정답률:  1.0
# LabelPropagation 의 정답률:  1.0
# LabelSpreading 의 정답률:  1.0
# LinearDiscriminantAnalysis 의 정답률:  1.0
# LinearSVC 의 정답률:  1.0
# LogisticRegression 의 정답률:  1.0
# LogisticRegressionCV 의 정답률:  1.0
# MLPClassifier 의 정답률:  1.0
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률:  0.7
# NearestCentroid 의 정답률:  1.0
# NuSVC 의 정답률:  1.0
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률:  1.0
# Perceptron 의 정답률:  1.0
# QuadraticDiscriminantAnalysis 의 정답률:  1.0
# RadiusNeighborsClassifier 의 정답률:  0.5333333333333333
# RandomForestClassifier 의 정답률:  1.0
# RidgeClassifier 의 정답률:  0.9666666666666667
# RidgeClassifierCV 의 정답률:  1.0
# SGDClassifier 의 정답률:  1.0
# SVC 의 정답률:  1.0
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈