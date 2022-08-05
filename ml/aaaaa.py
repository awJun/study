import numpy as np
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split,KFold, cross_val_score 
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets = load_iris()

x = datasets['data']
y = datasets.target


print(x.shape, y.shape) 
print("y의 라벨값(y의 고유값)", np.unique(y)) #y의 라벨값(y의 고유값) [0 1 2]


x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)

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
    