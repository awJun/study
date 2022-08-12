"""
[핵심]
각 모델 성능 테스트  (해당 모델들은 회기 모델들이다.)
model_1 = LinearSVR()
model_2 = Perceptron()
model_3 = LinearRegression()
model_4 = KNeighborsRegressor()
model_5 = DecisionTreeRegressor()
model_6 = RandomForestRegressor()

neighbors, tree, ensemble 모델 정리
from sklearn.neighbors   # 이웃하는 ..? 검색하자
from sklearn.tree        # 더 좋은 것을 아래로 뿌리를 내린다(가지치기) 결정나무
from sklearn.ensemble    # 같이 넣었을 때 더 좋은 것을 캐치

[중요]
LogisticRegression
- 이것은 Regression들어가지만 분류 모델이다.
- LinearRegression 이친구가 회기 모델이다.
이 친구 빼고는 나머지는 다 Regression이 들어가면 회기 모델로 생각하면 된다.

Classifier가 들어가면 분류 모델로 생각하면 된다.

"""

#### 과제 2 
# activation : sigmoid, relu, linear 넣고 돌리기
# metrics 추가
# EarlyStopping 넣고
# 성능 비교
# 감상문, 느낀점 2줄이상!!!

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, LinearSVR
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from sklearn.datasets import load_diabetes
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=72
                                                    )
'''
print(x)
print(y)
print(x.shape, y.shape) # (506, 13) (506,)
print(datasets.feature_names) #싸이킷런에만 있는 명령어
print(datasets.DESCR)
'''

#2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression  #LogisicRegression 분류
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn. ensemble import RandomForestClassifier, RandomForestRegressor

### [ 방법1 ]#############################################################################################
# model = Perceptron(),LinearSVR(),LinearRegression(),KNeighborsRegressor(),DecisionTreeRegressor(),RandomForestRegressor()


# for i in model:    
#     model = i
    

#     #3. 컴파일, 훈련
#     model.fit(x_train, y_train)


#     #4. 평가, 예측

#     result = model.score(x_test,y_test)   

#     y_predict = model.predict(x_test)

#     print(f"{i} : ", round(result,4))

### [ 방법2 ]################################################################################################
model_1 = LinearSVR()
model_2 = Perceptron()
model_3 = LinearRegression()
model_4 = KNeighborsRegressor()
model_5 = DecisionTreeRegressor()
model_6 = RandomForestRegressor()


#2. 컴파일, 훈련
model_1.fit(x_train, y_train)
model_2.fit(x_train, y_train)
model_3.fit(x_train, y_train)
model_4.fit(x_train, y_train)
model_5.fit(x_train, y_train)
model_6.fit(x_train, y_train)


#4. 평가, 예측
from sklearn.metrics import r2_score
y_predict_1 = model_1.predict(x_test)
y_predict_2 = model_2.predict(x_test)
y_predict_3 = model_3.predict(x_test)
y_predict_4 = model_4.predict(x_test)
y_predict_5 = model_5.predict(x_test)
y_predict_6 = model_6.predict(x_test)

acc_1 = r2_score(y_test, y_predict_1)
acc_2 = r2_score(y_test, y_predict_2)
acc_3 = r2_score(y_test, y_predict_3)
acc_4 = r2_score(y_test, y_predict_4)
acc_5 = r2_score(y_test, y_predict_5)
acc_6 = r2_score(y_test, y_predict_6)
print('LinearSVC_r2_score : ', acc_1)
print('Perceptron_r2_score : ', acc_2)
print('LogisticRegression_r2_score : ', acc_3)
print('KNeighborsClassifier_r2_score : ', acc_4)
print('DecisionTreeClassifier_r2_score : ', acc_5)
print('RandomForestClassifier_r2_score : ', acc_6)

results_1 = model_1.score(x_test, y_test)
results_2 = model_2.score(x_test, y_test)
results_3 = model_3.score(x_test, y_test)
results_4 = model_4.score(x_test, y_test)
results_5 = model_5.score(x_test, y_test)
results_6 = model_6.score(x_test, y_test)
print("LinearSVC_결과 r2_score : ", results_1)   # 회기는 r2 / 분류는 acc로 결과가 나온다.
print("Perceptron_결과 r2_score : ", results_2)  
print("LogisticRegression_결과 r2_score : ", results_3)  
print("KNeighborsClassifier_결과 r2_score : ", results_4)  
print("DecisionTreeClassifier_결과 r2_score : ", results_5)  
print("RandomForestClassifier_결과 r2_score : ", results_6)  


# LinearSVC_r2_score :  -0.4485536197256561
# Perceptron_r2_score :  0.4314332067390366
# LogisticRegression_r2_score :  0.6579209558684551
# KNeighborsClassifier_r2_score :  0.5403351561734346
# DecisionTreeClassifier_r2_score :  0.08384139107734101
# RandomForestClassifier_r2_score :  0.5654669291011905

# LinearSVC_결과 r2_score :  -0.4485536197256561
# Perceptron_결과 r2_score :  0.011235955056179775
# LogisticRegression_결과 r2_score :  0.6579209558684551
# KNeighborsClassifier_결과 r2_score :  0.5403351561734346
# DecisionTreeClassifier_결과 r2_score :  0.08384139107734101
# RandomForestClassifier_결과 r2_score :  0.5654669291011905


