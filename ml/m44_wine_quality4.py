import pandas as pd
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

path = "D:/study_data/_data/"

datasets = pd.read_csv(path+'winequality-white.csv',index_col=None, header=0, sep=';')

# print(datasets.shape)  # (4898, 12)
# print(datasets.describe())   # pandas에서만 사용가능
# print(datasets.info())   # 결측치 확인

import numpy as np  # 조금 더 빠르다 ..? 뭐지..
#--[ 둘 다 같은 방법임]--------
datasets2 = datasets.values
# datasets = datasets.to_numpy()     # numpy는 인덱스와 컬럼이 없다.
#---------------------------------
# x = datasets.drop(['quality'], axis=1)
# y = datasets['quality']
x = datasets2[:, :11]   # 11째 까지
y = datasets2[:, 11]    # 11 위치에 잇는 거

print(x.shape, y.shape)  # (4898, 11) (4898,)

# print(np.unique(y, return_counts=True))
# # (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# # 분류에서는 다중이든 이진이든 확인할것

# print(datasets["quality"].value_counts())






# 이걸 참고해서 아래 if, elif, else 조건문에 줄일 컬럼을 제한 걸것
# print(np.unique(y, return_counts=True))
# # (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))


# print(y[:10])   # y 값을 앞에서부터 10개만 보여줘라.
# # [6. 6. 6. 6. 6. 6. 6. 6. 6. 6.]   6이 너무 많네;;; 역시!  갯수가 2198이라서 그런거야 


####[ 메인 ]##########################################################################################################

# y 라벨을 줄일거임 이거는 업무에서만 사용하셈 회사에서는 오더한테 허락받아야하고 캐글에서는 사용불가!##
# print(y[:20]) 
# [6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 5. 5. 5. 7. 5. 7. 6. 8. 6. 5.]   # i 안에  6 ~ 5까지 순차적으로 들어감
newlist = []
for i in y:  # y를 다 넣음   y 범위 3 ~ 9
    if i <=5:
        newlist += [0]    # 
    elif i==6:
        newlist += [1]    # 
    else:
        newlist += [2]    # 
        
print(np.unique(newlist, return_counts=True))    # (array([0, 1, 2]), array([1640, 2198, 1060], dtype=int64))


########################################################################################################################






from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, newlist,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=123,
                                                    )

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

score = model.score(x_test, y_test)
print("model.score : ", score)

from sklearn.metrics import accuracy_score, f1_score
# print('acc_score : ', accuracy_score(y_test, y_predict))
# print("f1_score(macro) : ", f1_score(y_test, y_predict, average='macro'))
print("f1_score(micro) : ", f1_score(y_test, y_predict, average='micro'))




# ###[ 기본형태 ] ######################

# Name: quality, dtype: int64
# model.score :  0.6979591836734694
# f1_score(micro) :  0.6979591836734694

# ###[ y컬럼 줄인 후 ] ##################

# model.score :  0.7295918367346939
# f1_score(micro) :  0.7295918367346939

