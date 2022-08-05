from unittest import result
from sklearn.svm import LinearSVC, LinearSVR                   
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston   # sklearn은 학습 예제가 많음

#1. 데이터
datasets = load_boston()   
x = datasets.data    # 데이터가
y = datasets.target  # y에 들어간다.

print(x)
print(y)

print(x.shape, y.shape) # (506, 13) (506,)  열 13    (506, ) 506개 스칼라, 1개의 백터
                        # intput (506, 13), output 1
print(datasets.feature_names)
 # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 # 'B' 'LSTAT']
 
print(datasets.DESCR)
 

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=68
)



#2. 모델구성
model = LinearSVR()   # LinearSVC 이건 수치  /  LinearSVR 이건 분류

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

result = model.score(x_test, y_test)
print("결과 : ", round(result))

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)

# acc = accuracy_score(y_test, y_predict)
# print('accuracy : ', acc)

# results = model.score(x_test, y_test)
# print("결과 acc : ", results)   # 회기는 r2 / 분류는 acc로 결과가 나온다.


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)   

# print('loss : ', loss)
print('r2스코어 : ', r2)


