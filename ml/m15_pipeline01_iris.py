

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=1234
                                                    )

# scaler = MinMaxScaler()         # 파이프 라인에서 선언할것이므로 주석처리
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline  # 스케일링을 파이프 라인으로 넘겨주기 위해 import함

# model = SVC()   #파이프 라인을 쓸 것이므로 주석처리
model = make_pipeline(MinMaxScaler(), SVC())  # 스케일은 minmax 모델은 SVC를 사용하겟다

#3. 훈련
model.fit(x_train, y_train)   # 위에서 모델에서 make_pipeline를 했으므로 여기서 scaler.fit_transform과 함께 같이 수행한다       .

#4. 평가, 예측
result = model.score(x_test, y_test)

print("model.score : ", result)
# model.score :  1.0






