# n_component > 0.95 이상
# xgboost, gridSearch 또는 RandomSearch를 쓸 것

# m27_2의 결과를 뛰어넘어랏!!


# Parameter = [
#     {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.001, 0.01],
#      "max_depth":[4, 5, 6]},
#     {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01],
#      "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1]},
#     {"n_estimators":[90, 110], "learning_rate":[0.1, 0.001, 0.01],
#      "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9, 1],
#     "colsample_bylevel":[0.6, 0.7, 0.9]}
# ]
# n_jobs = -1
# tree_method = "gpu_hist", predictor="gpu_predictor", gpu_id=0,

# [실습시작]


# n_component > 0.95
# xgboost, gridSearch 또는 RandomSearch를 쓸 것

# n_jobs = -1
# tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1

"""

LDA = 따로 import 안하고 LinearDiscriminantAnalysis로 사용한다.

[주의점]
n_components 사용할 때 y의 라벨보다 큰수를 넣으면 안된다
라벨 갯수보다 많으면 ValueError: n_components cannot be larger than min(n_features, n_classes - 1). 발생

"""

import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import time
import warnings
warnings.filterwarnings(action='ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x = np.append(x_train, x_test, axis=0)

# print(x.shape)    # (60000, 28, 28)  -->  (70000, 784)

# print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6], dtype=int64), array([169472, 226640,  28603,   2198,   7594,  13894,  16408],
#       dtype=int64))


parameters = [
{'n_estimators':[100,200,300], 'learning_rate':[0.1, 0.3, 0.001, 0.01], 'max_depth':[4,5,6]},
{'n_estimators':[90,100,110], 'learning_rate':[0.1, 0.001, 0.01], 'max_depth':[4,5,6], 'colsample_bytree':[0.6,0.9,1]},
{'n_estimators':[90,110], 'learning_rate':[0.1, 0.001, 0.5], 'max_depth':[4,5,6],'colsample_bytree':[0.6,0.7,0.9]}
]

model = RandomizedSearchCV(XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', 
                                         gpu_id=1), parameters, verbose=1, refit=True, n_jobs=-1)



# lda = LinearDiscriminantAnalysis(n_components=6)
# lda.fit(x_train, y_train) 
# x_train = lda.transform(x_train)
# x_test = lda.transform(x_test)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
a=[]
pca = PCA(n_components=x_train.shape[1])
x = LinearDiscriminantAnalysis.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
cumsum = np.cumsum(pca_EVR)
a.append(np.argmax(cumsum >= 0.95)+1)
a.append(np.argmax(cumsum >= 0.99)+1)
a.append(np.argmax(cumsum >= 0.999)+1)
a.append(np.argmax(cumsum >= 1.0)+1)
print(a)

# model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=1)
# for i in range(len(a)):
#     n = a[i]
#     pca = PCA(n_components=n)
#     x2_train = pca.fit_transform(x_train)
#     x2_test = pca.transform(x_test)
#     start = time.time()
#     model.fit(x2_train, y_train, verbose=True)
#     end = time.time()
    
#     results = model.score(x2_test, y_test)
#     print(n, '의 결과: ', results)
#     print('시간: ', end-start)

# # # 3. 훈련
# # start = time.time()
# # model.fit(x_train, y_train)
# # end = time.time()


# # results = model.score(x_test, y_test)
# # print('결과: ', results)
# # print('시간: ', end-start)



# # PCA 0.95 이상
# # 결과:  0.9649
# # 시간:  510.00438714027405

# # PCA 0.99 이상
# # 결과:  0.9307
# # 시간:  527.3115980625153


# # 결과:  0.8691
# # 시간:  248.48112535476685


