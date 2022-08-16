import pandas as pd
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

path = "D:/study_data/_data/"

datasets = pd.read_csv(path+'winequality-white.csv',index_col=None, header=0, sep=';')

# print(datasets.head(5))    # (4898, 12)

# print(datasets.isnull().sum()) # 결측치 없는 것으로 판명

x = datasets.drop(['quality'], axis=1)
y = datasets['quality']

# print(x.head(5))
# print(y.head(5))

from sklearn.preprocessing import LabelEncoder   # XGBboost에서는 사용해야 하는 거 같음 저번에는 안그랬다고함..ㅠ
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import StratifiedKFold, KFold, HalvingGridSearchCV

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=9)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'n_jobs':[-1,2,4]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10], 'n_jobs':[-1,2,4]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10], 'n_jobs':[-1,2,4]},
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
    ] 




#2. 모델구성
# from xgboost import XGBClassifier
# model = XGBClassifier(random_state=123,
#                       n_estimators=1000,
#                       learning_rate=0.1,
#                       max_depth=3,
#                       gamma=1,
#                     )

from sklearn.ensemble import RandomForestClassifier
model = HalvingGridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)


# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=100)
# model = RandomizedSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)


# 3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()




from sklearn.metrics import accuracy_score
ypred = model.predict(x_test)
print('acc score: ', accuracy_score(y_test, ypred))
print('걸린시간: ', round(end-start,2), '초')



# acc score:  0.9979591836734694
# 걸린시간:  173.28 초



# print('최적의 매개변수: ', model.best_estimator_)
# print('최적의 파라미터: ', model.best_params_)
# print('best_score_: ', model.best_score_)
# print('model.score: ', model.score(x_test, y_test))
# ypred = model.predict(x_test)
# print('acc score: ', accuracy_score(y_test, ypred))
# ypred_best = model.best_estimator_.predict(x_test)
# print('best tuned acc: ', accuracy_score(y_test, ypred_best))

# print('걸린시간: ', round(end-start,2), '초')





# best_score_:  0.9941756838308562
# model.score:  0.9918367346938776
# acc score:  0.9918367346938776
# best tuned acc:  0.9918367346938776
# 걸린시간:  40.04 초





