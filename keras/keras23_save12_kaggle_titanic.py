from pydoc import describe
import numpy as np
import pandas as pd   # https://pandas.pydata.org/docs/index.html pandas 종합 설명 사이트

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import time

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




scaler =  MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) #
test_set = scaler.transform(test_set) # 
 
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 1.0000000000000002
# print(np.min(x_test))   # -0.06141956477526944
# print(np.max(x_test))   # 1.1478180091225068





#2. 모델 구성
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
"""
### 기존 모델 ###

model = Sequential()
model.add(Dense(100,input_dim=9))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#다중 분류로 나오는 아웃풋 노드의 개수는 y 값의 클래스의 수와 같다.활성화함수 'softmax'를 통해 
# 아웃풋의 합은 1이 된다.
"""
### 새로운 모델 ###
input1 = Input(shape=(9,))   # 처음에 Input 명시하고 Input 대한 shape 명시해준다.
dense1 = Dense(100)(input1)   # Dense 구성을하고  node 값을 넣고 받아오고 싶은 변수 받아온다.
dense2 = Dense(100, activation = 'relu')(dense1)    # 받아온 변수를 통해 훈련의 순서를 사용자가 원하는대로 할 수 있다.
dense3 = Dense(100, activation = 'sigmoid')(dense2)
output1 = Dense(1, activation='sigmoid')(dense3)
model = Model(inputs=input1, outputs=output1) # 해당 모델의 input과 output을 설정한다.



#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=200, mode='min', 
                              verbose=1,restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=20, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2
                )
end_time = time.time() -start_time

#다중 분류 모델은 'categorical_crossentropy'만 사용한다 !!!!


model.save("./_save/keras23_12_save_kaggle_titanic.h5")



#4.  평가,예측

loss,acc = model.evaluate(x_test,y_test)
print('loss :',loss)
# print('accuracy :',acc)
# print("+++++++++  y_test       +++++++++")
# print(y_test[:5])
# print("+++++++++  y_pred     +++++++++++++")
# result = model.evaluate(x_test,y_test) 위에와 같은 개념 [0] 또는 [1]을 통해 출력가능
# print('loss :',result[0])
# print('accuracy :',result[1])




y_predict = model.predict(x_test)
y_predict[(y_predict<0.5)] = 0  
y_predict[(y_predict>=0.5)] = 1  
print(y_predict) 
print(y_test.shape) #(134,)


# y_test = np.argmax(y_test,axis=1)
# import tensorflow as tf
# y_test = np.argmax(y_test,axis=1)
# y_predict = np.argmax(y_predict,axis=1)
#pandas 에서 인코딩 진행시 argmax는 tensorflow 에서 임포트한다.
# print(y_test.shape) #(87152,7)
# y_test와 y_predict의  shape가 일치해야한다.



acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)


y_summit = model.predict(test_set)

gender_submission['Survived'] = y_summit
submission = gender_submission.fillna(gender_submission.mean())
submission [(submission <0.5)] = 0  
submission [(submission >=0.5)] = 1  
submission = submission.astype(int)
submission.to_csv('test21.csv',index=True)


print("걸린시간 : ", end_time)











# model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 100)               1000
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               10100
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 101
# =================================================================
# Total params: 11,201
# Trainable params: 11,201
# Non-trainable params: 0
# _________________________________________________________________



#########################################################
"""   [best_scaler]

scaler =  MinMaxScaler()

loss : 0.4683985710144043
acc 스코어 : 0.7901234567901234
걸린시간 :  4.877824306488037
"""
#########################################################
"""   
scaler 사용 안함

loss : 0.8532899618148804
acc 스코어 : 0.6790123456790124
걸린시간 :  13.260047435760498
"""
#########################################################
"""
scaler = StandardScaler()

loss : 0.44449031352996826
acc 스코어 : 0.8148148148148148
걸린시간 :  13.54603624343872
"""
#########################################################
"""
scaler =  MinMaxScaler()

loss : 0.4293496012687683
acc 스코어 : 0.8148148148148148
걸린시간 :  13.505566835403442
"""
#########################################################
"""
scaler = MaxAbsScaler()

loss : 0.4385797083377838
acc 스코어 : 0.8148148148148148
걸린시간 :  12.953725576400757
"""
#########################################################
"""
scaler = RobustScaler()

loss : 0.5573062300682068
acc 스코어 : 0.8024691358024691
걸린시간 :  13.394407272338867
"""  
#########################################################
 










































