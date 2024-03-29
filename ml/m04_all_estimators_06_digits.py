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

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

# 1. 데이터
datasets = load_digits()
print(datasets.DESCR)
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

# AdaBoostClassifier 의 정답률:  0.2972222222222222
# BaggingClassifier 의 정답률:  0.9305555555555556
# BernoulliNB 의 정답률:  0.8333333333333334
# CalibratedClassifierCV 의 정답률:  0.9638888888888889
# CategoricalNB 은 안나온 놈
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률:  0.7972222222222223
# DecisionTreeClassifier 의 정답률:  0.8305555555555556
# DummyClassifier 의 정답률:  0.08055555555555556
# ExtraTreeClassifier 의 정답률:  0.7833333333333333
# ExtraTreesClassifier 의 정답률:  0.9888888888888889
# GaussianNB 의 정답률:  0.8277777777777777
# GaussianProcessClassifier 의 정답률:  0.9833333333333333
# GradientBoostingClassifier 의 정답률:  0.9666666666666667
# HistGradientBoostingClassifier 의 정답률:  0.9777777777777777
# KNeighborsClassifier 의 정답률:  0.9888888888888889
# LabelPropagation 의 정답률:  0.9888888888888889
# LabelSpreading 의 정답률:  0.9888888888888889
# LinearDiscriminantAnalysis 의 정답률:  0.9527777777777777
# LinearSVC 의 정답률:  0.9666666666666667
# LogisticRegression 의 정답률:  0.9583333333333334
# LogisticRegressionCV 의 정답률:  0.9666666666666667
# MLPClassifier 의 정답률:  0.9694444444444444
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률:  0.8805555555555555
# NearestCentroid 의 정답률:  0.8888888888888888
# NuSVC 의 정답률:  0.9555555555555556
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률:  0.9222222222222223
# Perceptron 의 정답률:  0.9527777777777777
# QuadraticDiscriminantAnalysis 의 정답률:  0.8666666666666667
# RadiusNeighborsClassifier 은 안나온 놈
# RandomForestClassifier 의 정답률:  0.975
# RidgeClassifier 의 정답률:  0.95
# RidgeClassifierCV 의 정답률:  0.95
# SGDClassifier 의 정답률:  0.9527777777777777
# SVC 의 정답률:  0.9888888888888889
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈