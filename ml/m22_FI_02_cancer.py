# 실습
# 피처임포턴스가 전체 중요도에서 하위 20~25% 칼럼들을 제거하여
# 데이터셋 재구성 후
# 각 모델별로 돌려서 결과 도출

# 기존 모델결과와 비교

# 결과비교
# 1. DecisionTree
# 기존 acc: 
# 칼럼삭제 후 acc:
"""
[핵심]
if str(model).startswith('XGB'):     # startswith('XGB') : XGB로 이름이 시작하면 ! 이라는 조건을 걸게해줌

featurelist.append(np.argsort(model.feature_importances_)[a])   # argsort : 리스트 정렬

print(str(model).strip('()'), '의 드랍후 스코어: ', score)   # strip('()') 공백과 ()를 제거
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))


x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

# 2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

# 3. 컴파일, 훈련, 평가, 예측
for model in models:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if str(model).startswith('XGB'):   # 만약 str(model)이 XGB로 시작하면
        print('XGB 의 스코어:        ', score)
    else:
        print(str(model).strip('()'), '의 스코어:        ', score)
        
    featurelist = []
    for a in range(int(allfeature)):
        featurelist.append(np.argsort(model.feature_importances_)[a])
        
    x_bf = np.delete(x, featurelist, axis=1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_bf, y, shuffle=True, train_size=0.8, random_state=1234)
    model.fit(x_train2, y_train2)
    score = model.score(x_test2, y_test2)
    if str(model).startswith('XGB'):
        print('XGB 의 드랍후 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 드랍후 스코어: ', score)

# 자를 갯수:  6
# DecisionTreeClassifier 의 스코어:         0.8947368421052632
# DecisionTreeClassifier 의 드랍후 스코어:  0.8859649122807017
# RandomForestClassifier 의 스코어:         0.9298245614035088
# RandomForestClassifier 의 드랍후 스코어:  0.9298245614035088
# GradientBoostingClassifier 의 스코어:         0.9122807017543859
# GradientBoostingClassifier 의 드랍후 스코어:  0.9122807017543859
# XGB 의 스코어:         0.9385964912280702
# XGB 의 드랍후 스코어:  0.9385964912280702