from pydoc import describe
import numpy as np
import pandas as pd   # https://pandas.pydata.org/docs/index.html pandas 종합 설명 사이트
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM
from tensorflow.python.keras.models import Sequential
from keras.datasets import mnist,cifar10,cifar100,fashion_mnist
import tensorflow as tf       
import time
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.datasets import load_wine





#1.데이터

path = './_data/kaggle_titanic/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

"""
print(train_set) # [891 rows x 11 columns]

             Survived  Pclass  ... Cabin Embarked
PassengerId                    ...
1                   0       3  ...   NaN        S   
2                   1       1  ...   C85        C   
3                   1       3  ...   NaN        S   
4                   1       1  ...  C123        S   
5                   0       3  ...   NaN        S   
...               ...     ...  ...   ...      ...   
887                 0       2  ...   NaN        S   
888                 1       1  ...   B42        S   
889                 0       3  ...   NaN        S   
890                 1       1  ...  C148        C   
891                 0       3  ...   NaN        Q   

[891 rows x 11 columns]
"""
################################################################################
"""
print(train_set.describe())  # describe 해당 데이터에 상세 내용을 볼 수 있다.

https://www.geeksforgeeks.org/python-pandas-dataframe-describe-method/    ## describe 관련 링크 ##

         Survived  ...        Fare
count  891.000000  ...  891.000000
mean     0.383838  ...   32.204208
std      0.486592  ...   49.693429
min      0.000000  ...    0.000000
25%      0.000000  ...    7.910400
50%      0.000000  ...   14.454200
75%      1.000000  ...   31.000000
max      1.000000  ...  512.329200

[8 rows x 6 columns]
"""
################################################################################
"""
print(train_set.info())

 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   Survived  891 non-null    int64
 1   Pclass    891 non-null    int64
 2   Name      891 non-null    object
 3   Sex       891 non-null    object
 4   Age       714 non-null    float64
 5   SibSp     891 non-null    int64
 6   Parch     891 non-null    int64
 7   Ticket    891 non-null    object
 8   Fare      891 non-null    float64
 9   Cabin     204 non-null    object
 10  Embarked  889 non-null    object
"""
################################################################################
"""
print(test_set) # [418 rows x 10 columns]


print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계

# Survived      0
# Pclass        0
# Name          0
# Sex           0
# Age         177
# SibSp         0
# Parch         0
# Ticket        0
# Fare          0
# Cabin       687
# Embarked      2
"""
################################################################################
"""
# train_set = train_set.fillna(train_set.median())


print(test_set.isnull().sum())

# Pclass        0
# Name          0
# Sex           0
# Age          86
# SibSp         0
# Parch         0
# Ticket        0
# Fare          1
# Cabin       327
# Embarked      0
"""
drop_cols = ['Cabin']
train_set.drop(drop_cols, axis = 1, inplace =True)  # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html  Drop 링크 
test_set = test_set.fillna(test_set.mean())   # https://www.statology.org/pandas-fillna-with-mean/  결측치 평균으로 채우는 방법
train_set['Embarked'].fillna('S')
train_set = train_set.fillna(train_set.mean())

print(train_set) 
print(train_set.isnull().sum())

test_set.drop(drop_cols, axis = 1, inplace =True) # https://www.geeksforgeeks.org/what-does-inplace-mean-in-pandas/
               # inplace의 디폴트는 false  /  True로 지정하면  개체의 복사본을 반환합니다.
cols = ['Name','Sex','Ticket','Embarked'] 
for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
x = train_set.drop(['Survived'],axis=1) #axis는 컬럼 
print(x) #(891, 9)
y = train_set['Survived']
print(y.shape) #(891,)


# test_set.drop(drop_cols, axis = 1, inplace =True)
gender_submission = pd.read_csv(path + 'gender_submission.csv',#예측에서 쓸거야!!
                       index_col=0)
# y의 라벨값 : (array([0, 1], dtype=int64), array([549, 342], dtype=int64))

###########(pandas 버전 원핫인코딩)###############
# y_class = pd.get_dummies((y))
# print(y_class.shape) # (891, 2)

# 해당 기능을 통해 y값을 클래스 수에 맞는 열로 늘리는 원핫 인코딩 처리를 한다.
#1개의 컬럼으로 [0,1,2] 였던 값을 ([1,0,0],[0,1,0],[0,0,1]과 같은 shape로 만들어줌)



x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.91,
                                                    shuffle=True,
                                                    random_state=100)
#셔플을 False 할 경우 순차적으로 스플릿하다보니 훈련에서는 나오지 않는 값이 생겨 정확도가 떨어진다.
#디폴트 값인  shuffle=True 를 통해 정확도를 올린다.

# - -[ 긁어온거 끝 ~ ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 



#--[해당 데이터의 unique형태 확인]- - - - - - - - - - - - - - - - - 
# print(np.unique(x_train, return_counts=True))
# print(np.unique(y_train, return_counts=True))    # <--- 이 항목은 확인 필수 (이진분류)
# print(np.unique(x_test, return_counts=True))   
# print(np.unique(y_test, return_counts=True))
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#--[해당 데이터 shape 확인]- - - - - - - - - - - - - - - - - - - - -
# print(x_train.shape)   # (810, 9)
# print(y_train.shape)   # (810,
# print(x_test.shape)    # (81, 9)
# print(y_test.shape)    # (81,)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#--[ 스케일러 작업]- - - - - - - - - - - - - - - -(중간값을 찾아주는 역할 (값들의 차이 완화)
# scaler =  MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) 
# #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# #--[차원 변형 작업]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
x_train = x_train.reshape(810, 3, 3)               
x_test = x_test.reshape(81, 3, 3)

# print(x_train.shape)  # (810, 3, 3, 1)     <-- "3, 2 ,1"는 input_shape값
# print(x_test.shape)   # (81, 3, 3, 1)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#- -[y데이터를 x와 동일한 차원 형태로 변환] - - - - - - - - - - - ( [중요]!! 회귀형에서는 할 필요없음  )
from tensorflow.python.keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) # (810, 2) (81, 2)


#2. 모델구성
model = Sequential()
model.add(LSTM(units=100 ,input_length=3, input_dim=3))   #SimpleRNN  또는 LSTM를 거치면 3차원이 2차원으로 변형되어서 다음 레어어에 간다.
# model.add(SimpleRNN(10))       # <-- 3차원이 아니라 2차원을 넣어서 에러가 발생함.[참고]
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))   
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax')) # sigmoid에선 output은 1이다. (softmax에서는 유니크 갯수만큼)
model.summary()


#3. 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=1000, mode='auto', verbose=1, 
                              restore_best_weights=True)   
start_time = time.time()
model.fit(x_train, y_train, epochs=10000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[es],
                 verbose=1)
end_time = time.time() -start_time



#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test) 

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis= 1)  # 위에서 원핫을 했으므로 argmax로 평가지표를 사용할 수 있게 만듬
y_test = np.argmax(y_test, axis= 1)

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)



print('loss : ', loss)
print('acc : ', acc)  
print("RMSE : ", rmse)
print("걸린시간:", end_time )

# loss :  0.4894302189350128
# acc :  0.790123462677002
# RMSE :  0.4581228472908512
# 걸린시간: 52.87075924873352


























#-----------------------------------------------------------------------









# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)



# - - - - - - - - - - - - - - - - - - - - - -
# acc = accuracy_score(y_test, y_predict)
# print('acc 스코어 :', acc)


# y_summit = model.predict(test_set)

# gender_submission['Survived'] = y_summit
# submission = gender_submission.fillna(gender_submission.mean())
# submission [(submission <0.5)] = 0  
# submission [(submission >=0.5)] = 1  
# submission = submission.astype(int)
# submission.to_csv('test21.csv',index=True)
#-- - - - - - - - - - - - - - - - - - - - - - - 

































