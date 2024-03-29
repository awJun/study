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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.utils import all_estimators
import warnings
from sklearn.datasets import fetch_covtype
warnings.filterwarnings('ignore') # warnig 출력 안함

# 1. 데이터
datasets = fetch_covtype()
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

# AdaBoostClassifier 의 정답률:  0.5390222283417812
# BaggingClassifier 의 정답률:  0.962324552722391
# BernoulliNB 의 정답률:  0.6318081288779119
# CalibratedClassifierCV 의 정답률:  0.7133464712614993
# CategoricalNB 의 정답률:  0.6330989733483645
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률:  0.6215760350421246
# DecisionTreeClassifier 의 정답률:  0.9409137457724843
# DummyClassifier 의 정답률:  0.4869925905527396
# ExtraTreeClassifier 의 정답률:  0.8718707778628779
# ExtraTreesClassifier 의 정답률:  0.9533316695782381
# GaussianNB 의 정답률:  0.09068612686419456
# GaussianProcessClassifier 은 안나온 놈
# GradientBoostingClassifier 의 정답률:  0.7726134437148783
# HistGradientBoostingClassifier 의 정답률:  0.8356754989113878
# KNeighborsClassifier 의 정답률:  0.9372735643658081
# LabelPropagation 은 안나온 놈
# LabelSpreading 은 안나온 놈
# LinearDiscriminantAnalysis 의 정답률:  0.681092570759791
# LinearSVC 의 정답률:  0.7132432037038631
# LogisticRegression 의 정답률:  0.7204030876999733
# LogisticRegressionCV 의 정답률:  0.7249554658657694
# MLPClassifier 의 정답률:  0.8438680584838602
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률:  0.642401659165426
# NearestCentroid 의 정답률:  0.38807087596705764
# NuSVC 은 안나온 놈
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률:  0.6348889443473921
# Perceptron 의 정답률:  0.6159307418913453
# QuadraticDiscriminantAnalysis 의 정답률:  0.12789687013244064
# RadiusNeighborsClassifier 은 안나온 놈
# RandomForestClassifier 의 정답률:  0.9554228376203713
# RidgeClassifier 의 정답률:  0.7020730962195468
# RidgeClassifierCV 의 정답률:  0.7020558849599408
# SGDClassifier 의 정답률:  0.710231233272807