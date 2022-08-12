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

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.svm import LinearSVC, LinearSVR



#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (891, 11)
# print(train_set.describe())
# print(train_set.columns)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
# print(test_set)
# print(test_set.shape) # (418, 10)
# print(test_set.describe())

print(train_set.Pclass.value_counts())

Pclass1 = train_set["Survived"][train_set["Pclass"] == 1].value_counts(normalize = True)[1]*100
Pclass2 = train_set["Survived"][train_set["Pclass"] == 2].value_counts(normalize = True)[1]*100
Pclass3 = train_set["Survived"][train_set["Pclass"] == 3].value_counts(normalize = True)[1]*100
print(f"Percentage of Pclass 1 who survived: {Pclass1}")
print(f"Percentage of Pclass 2 who survived: {Pclass2}")
print(f"Percentage of Pclass 3 who survived: {Pclass3}")

female = train_set["Survived"][train_set["Sex"] == 'female'].value_counts(normalize = True)[1]*100
male = train_set["Survived"][train_set["Sex"] == 'male'].value_counts(normalize = True)[1]*100
print(f"Percentage of females who survived: {female}")
print(f"Percentage of males who survived: {male}")

sns.barplot(x="SibSp", y="Survived", data=train_set)


# df = pd.DataFrame(y)
# print(df)
# oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
# y = oh.fit_transform(df)
# print(y)



# print(test_set.columns)
# print(train_set.info()) # info 정보출력
# print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

#### 결측치 처리 1. 제거 ####

train_set = train_set.fillna({"Embarked": "C"})
train_set.Age = train_set.Age.fillna(value=train_set.Age.mean())

train_set = train_set.drop(['Name'], axis = 1)
test_set = test_set.drop(['Name'], axis = 1)

train_set = train_set.drop(['Ticket'], axis = 1)
test_set = test_set.drop(['Ticket'], axis = 1)

train_set = train_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)

train_set = pd.get_dummies(train_set,drop_first=True)
test_set = pd.get_dummies(test_set,drop_first=True)

test_set.Age = test_set.Age.fillna(value=test_set.Age.mean())
test_set.Fare = test_set.Fare.fillna(value=test_set.Fare.mode())

print(train_set, test_set, train_set.shape, test_set.shape)

############################


x = train_set.drop(['Survived'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (891, 8)

y = train_set['Survived'] 
print(y)
print(y.shape) # (891,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0


#2. 모델구성

model = LinearSVC()
                                                                        
#3. 컴파일, 훈련
model.fit(x_train, y_train)



#4. 평가, 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)

print(y_predict)
y_predict = y_predict.round(0)
print(y_predict)

acc= accuracy_score(y_test, y_predict)
print('acc : ' , results)
print('acc스코어 : ', acc) 

y_summit = model.predict(test_set)

print(y_summit)
print(y_summit.shape) # (418, 1)
y_summit = y_summit.round()
df = pd.DataFrame(y_summit)
print(df)
oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
y_summit = oh.fit_transform(df)
print(y_summit)
y_summit = np.argmax(y_summit, axis= 1)
# submission_set = pd.read_csv(path + 'gender_submission.csv', # + 명령어는 문자를 앞문자와 더해줌
#                              index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

# print(submission_set)

# submission_set['Survived'] = y_summit
# print(submission_set)


# submission_set.to_csv(path + 'submission_maxabs.csv', index = True)



model.summary()

# acc :  0.770949720670391

# 스케일러 하기전 
# loss :  0.4480155110359192
# acc스코어 :  0.8268156424581006

# 민맥스
# loss :  [0.45806556940078735, 0.7988826632499695]
# acc스코어 :  0.7988826815642458

# 스탠다드
# loss :  [0.41274765133857727, 0.7932960987091064]
# acc스코어 :  0.7932960893854749

# maxabs
# loss :  [0.41841256618499756, 0.8044692873954773]
# acc스코어 :  0.8044692737430168
# robust
# loss :  [0.397939532995224, 0.8156424760818481]
# acc스코어 :  0.8156424581005587