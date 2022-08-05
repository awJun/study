import numpy as np
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split,KFold, cross_val_score 
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
              
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
allalgorithm = all_estimators(type_filter='classifier')

print('allalgorithms : ', allalgorithm)
print("모델의 갯수 : ", len(allalgorithm)) #모델의 갯수 :  41
print('kfold_all_estimators')
for (name, algorithm) in allalgorithm : #name-algorithm : key-value 쌍으로 이루는 dictionary
  try : 
      model = algorithm()
      model.fit(x_train, y_train)
 
      y_predict = model.predict(x_test)
      scores = cross_val_score(model,x,y,cv=kfold)
      print(name,
            # scores,'\ncross_val_score :', 
            round(np.mean(scores),4))
  except : 
    #   continue
    print(name, ": 미출력!!!!!!!!")


# kfold_all_estimators
# AdaBoostClassifier 0.8867
# BaggingClassifier 0.9667
# BernoulliNB 0.2933
# CalibratedClassifierCV 0.9133
# CategoricalNB 0.9333
# ClassifierChain : 미출력!!!!!!!!
# ComplementNB 0.6667
# DecisionTreeClassifier 0.9467
# DummyClassifier 0.2933
# ExtraTreeClassifier 0.9333
# ExtraTreesClassifier 0.9467
# GaussianNB 0.9467
# GaussianProcessClassifier 0.96
# GradientBoostingClassifier 0.9667
# HistGradientBoostingClassifier 0.94
# KNeighborsClassifier 0.96
# LabelPropagation 0.96
# LabelSpreading 0.96
# LinearDiscriminantAnalysis 0.98
# LinearSVC 0.9667
# LogisticRegression 0.9667
# LogisticRegressionCV 0.9733
# MLPClassifier 0.9667
# MultiOutputClassifier : 미출력!!!!!!!!
# MultinomialNB 0.9667
# NearestCentroid 0.9333
# NuSVC 0.9733
# OneVsOneClassifier : 미출력!!!!!!!!
# OneVsRestClassifier : 미출력!!!!!!!!
# OutputCodeClassifier : 미출력!!!!!!!!
# PassiveAggressiveClassifier 0.82
# Perceptron 0.78
# QuadraticDiscriminantAnalysis 0.98
# RadiusNeighborsClassifier 0.9533
# RandomForestClassifier 0.96
# RidgeClassifier 0.84
# RidgeClassifierCV 0.84
# SGDClassifier 0.86
# SVC 0.9667
# StackingClassifier : 미출력!!!!!!!!
# VotingClassifier : 미출력!!!!!!!!





