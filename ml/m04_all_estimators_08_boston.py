"""
all_estimators
 - 해당 데이터셋에서 사용할 수 있는 모든 알고리즘 모델들을 가져와준다.


사용법
Allalgorithm = all_estimators(type_filter='regressor')  # 회기 모델이므로 regressor
print("algorithm : ", Allalgorithm)
print("모델의 갯수 : ", len(Allalgorithm))   # 모델의 갯수 :  41  

 # 모델의 갯수 = 해당 데이터셋에서 사용할 수 있는 모델의 갯수를 뜻한다.


버전이 달라서 안돌아가는 모델들의 경우를 생각해서 예외처리 및 출력작업
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
from sklearn.datasets import load_boston
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')    
                           
#1. 데이터
datasets = load_boston()   
x = datasets.data    # 데이터가
y = datasets.target  # y에 들어간다.

print(x.shape, y.shape) # (506, 13) (506,)  열 13    (506, ) 506개 스칼라, 1개의 백터
                        # intput (506, 13), output 1
print(datasets.feature_names)
 # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 # 'B' 'LSTAT']

print(datasets.DESCR)

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


Allalgorithm = all_estimators(type_filter='regressor')
# print("algorithm : ", Allalgorithm)
# print("모델의 갯수 : ", len(Allalgorithm))   # 모델의 갯수 :  41

for (name, algorithm) in Allalgorithm:
    try:   # 예외처리
        model  = algorithm()
        model.fit(x_train, y_train)      # 훈련 및 컴파일
        
        y_predict = model.predict(x_test)
        acc = r2_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)  # 성능확인
    except:
        # continue    # continue : 계속 진행해
        print(name, '은 안나온 놈!!!')    # 안돌아가는 모델
        
        
        
# ARDRegression 의 정답률 :  0.7538021118752509
# AdaBoostRegressor 의 정답률 :  0.8551489609620271
# BaggingRegressor 의 정답률 :  0.8743203340295383
# BayesianRidge 의 정답률 :  0.7519553718541365
# CCA 의 정답률 :  0.7671481620320935
# DecisionTreeRegressor 의 정답률 :  0.6882657120194575
# DummyRegressor 의 정답률 :  -0.001983416409809813
# ElasticNet 의 정답률 :  0.10066848307211107
# ElasticNetCV 의 정답률 :  0.738830220441546
# ExtraTreeRegressor 의 정답률 :  0.7447410359307075
# ExtraTreesRegressor 의 정답률 :  0.8878613487476177
# GammaRegressor 의 정답률 :  0.13498106749480976
# GaussianProcessRegressor 의 정답률 :  -2.1699240819671215
# GradientBoostingRegressor 의 정답률 :  0.8968749656454579
# HistGradientBoostingRegressor 의 정답률 :  0.8596776043198935
# HuberRegressor 의 정답률 :  0.7284451206085558
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 :  0.7248521163487281
# KernelRidge 의 정답률 :  0.6986778129359407
# Lars 의 정답률 :  0.7412138114627917
# LarsCV 의 정답률 :  0.7420473049045087
# Lasso 의 정답률 :  0.17750664918670922
# LassoCV 의 정답률 :  0.7511199821186207
# LassoLars 의 정답률 :  -0.001983416409809813
# LassoLarsCV 의 정답률 :  0.753398553101863
# LassoLarsIC 의 정답률 :  0.7555033086871306
# LinearRegression 의 정답률 :  0.7555033086871308
# LinearSVR 의 정답률 :  0.49330883499851497
# MLPRegressor 의 정답률 :  0.3144995655372228
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 :  0.49142827229303754
# OrthogonalMatchingPursuit 의 정답률 :  0.5453164888101074
# OrthogonalMatchingPursuitCV 의 정답률 :  0.6978725032619453
# PLSCanonical 의 정답률 :  -1.8853216066612473
# PLSRegression 의 정답률 :  0.7210301120307101
# PassiveAggressiveRegressor 의 정답률 :  0.7208560090338274
# PoissonRegressor 의 정답률 :  0.4968289232825771
# RANSACRegressor 의 정답률 :  0.3525972518784318
# RadiusNeighborsRegressor 의 정답률 :  0.3175134599719306
# RandomForestRegressor 의 정답률 :  0.8807339564275026
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 :  0.72456494791284
# RidgeCV 의 정답률 :  0.7526706692256646
# SGDRegressor 의 정답률 :  0.6948221254790665
# SVR 의 정답률 :  0.48466515414059697
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 :  0.7221775544035689
# TransformedTargetRegressor 의 정답률 :  0.7555033086871308
# TweedieRegressor 의 정답률 :  0.13335700700988817
# VotingRegressor 은 안나온 놈!!!        