import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=1234
                                                    )

parameters = [
    {'RF__n_estimators':[100, 200],'RF__max_depth':[6, 8],'RF__min_samples_leaf':[3,5],
     'RF__min_samples_split':[2, 3]},
    {'RF__n_estimators':[300, 400],'RF__max_depth':[6, 8],'RF__min_samples_leaf':[7, 10],
     'RF__min_samples_split':[4, 7]}
    ]                                                                       

from sklearn.model_selection import KFold, StratifiedKFold
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)


# scaler = MinMaxScaler()         # 파이프 라인에서 선언할것이므로 주석처리
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline # 스케일링을 파이프 라인으로 넘겨주기 위해 import함
from sklearn.decomposition import PCA

# model = SVC()   #파이프 라인을 쓸 것이므로 주석처리
# model = make_pipeline(MinMaxScaler(), PCA(), RandomForestClassifier())  # 스케일은 minmax 모델은 SVC를 사용하겟다

Pipe = Pipeline([('minmax', MinMaxScaler()), ('RF', RandomForestClassifier())])  # RF은 변수형 RandomForestClassifier은 뭔제알겟제 ?
                                                                                 # 파라미터를 사용하면 파라미터 안에 변수를 넣어줘야한다.
#3. 훈련
#3. 훈련
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

model = GridSearchCV(Pipe, parameters, cv=5, verbose=1)

model.fit(x_train, y_train)   # 위에서 모델에서 make_pipeline를 했으므로 여기서 scaler.fit_transform과 함께 같이 수행한다       .

#4. 평가, 예측
# result = model.score(x_test, y_test)
# print("model.score : ", result)
# model.score :  0.9649122807017544

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)   # make_pipeline를 이용해서 스케일이 적용된다.
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)
