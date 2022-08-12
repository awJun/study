"""
[핵심]
# 통상적으로 컴퓨터가 좋으면 딥러닝 / 안좋으면 머신러닝을 활용한다.

머신러닝만 사용 할 것이므로 sklearn을 사용한다. 그러므로 tensorflow는 사용안할예정
머신러닝은 sklearn에 다 들어있다.

딥러닝은 레이어를 길게 뺀거
머신러닝은 간결해서 속도가 빠르다.

러닝머신은 원핫 할 필요없음 모델구성에서 알아서 받아짐
훈련에서 튜닝하고 평가할때 이벨류에이트없고 스코어를 사용한다.

LinearSVC
 - 분류모델에서 사용한다.
 - 하나의 레이어로 구성되어 있다. 

LinearSCR
 - 회기모델에서 사용한다.
 - 하나의 레이어로 구성되어 있다.

model.fit
 - model.fit(x_train, y_train)
 - 을 사용하면 fit 부분에서 컴파일까지 같이 자동으로 진행해줘서 여기서 fit과 compile이 같이된다.
 - 해당 방식은 러닝머신 모델에서만 사용이 가능하다.
 
model.score
 - results = model.score(x_test, y_test)  #분류 모델과 회귀 모델에서 score를 쓰면 알아서 자동으로 맞춰서 사용해준다. 
 - print("결과 acc : ", results)          # 회기는 r2 / 분류는 acc로 결과가 나온다.

[TMI]
러닝머신이 나온 이후 딥러닝이 나왔으므로 레이어에 대한 중요성을 몰랐을 때였다. 그때 만든 러닝머신 전용
모델인 LinearSVC, LinearSCR 는 레이어가 한 개인 모델로 만들어져있다. 이로 인해서 m03에서 배울 예정인
SVC, SCR이 만들어졌다. 이 모델은 레이어가 여러개이므로 m02의 Perceptron에서 해결못한 문제점을 해결했다.
"""

from unittest import result
from sklearn.svm import LinearSVC, LinearSVR    # LinearSVC 이건 수치  /  LinearSVR 이건 분류
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.datasets import fetch_california_housing
datasets = fetch_california_housing()

x=datasets.data
y=datasets.target

# print(x)
# [[   8.3252       41.            6.98412698 ...    2.55555556
#     37.88       -122.23      ]
#  [   8.3014       21.            6.23813708 ...    2.10984183
#     37.86       -122.22      ]
#  [   7.2574       52.            8.28813559 ...    2.80225989
#     37.85       -122.24      ]
#  ...
#  [   1.7          17.            5.20554273 ...    2.3256351
#     39.43       -121.22      ]
#  [   1.8672       18.            5.32951289 ...    2.12320917
#     39.43       -121.32      ]
#  [   2.3886       16.            5.25471698 ...    2.61698113
#     39.37       -121.24      ]]

# print(y)
# [4.526 3.585 3.521 ... 0.923 0.847 0.894]

print(x.shape, y.shape)   # x (20640, 8) 열이 8개   /  y (20640,)  

print(datasets.feature_names)  
 # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

print(datasets.DESCR)



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=68
                                                    )



#2. 모델구성
model = LinearSVR()

#3. 컴파일. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)    # 회기는 r2 / 분류는 acc로 결과가 나온다.

print("결과 : ", round(result))
print('r2스코어 : ', r2)



































