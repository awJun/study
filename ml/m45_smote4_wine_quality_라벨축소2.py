"""
[핵심]

# y 라벨을 줄일거임 이거는 업무에서만 사용하셈 회사에서는 오더한테 허락받아야하고 캐글에서는 사용불가!##

이걸로 안에 어떤 값이 들었는지 확인하고
# print(np.unique(y, return_counts=True))
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))

아래에서 조건문에서 걸어서 컬럼을 줄임

축소하기 전에 
print(np.unique(y, return_counts=True))
를 사용해서 먼저 y데이터를 확인하고

for index, value in enumerate(y):  # y를 다 넣음   y 범위 3 ~ 9
    if value == 9:    # 9.0        5
        y[index] = 7  
    elif value == 8:  # 8.0        175
        y[index] = 7
    elif value == 7:  # 7.0        880
        y[index] = 7
    elif value == 6:  # 6.0        2198
        y[index] = 6
    elif value == 5:  # 5.0        1457
        y[index] = 5
    elif value == 4:  # 4.0         163
        y[index] = 4
    elif value == 3:  # 3.0          20
        y[index] = 4
    else:
        y[index] = 0    

를 사용해서 y컬럼을 축소시키고
        
print(np.unique(y, return_counts=True))    # (array([4., 5., 6., 7.]), array([ 183, 1457, 2198, 1060], dtype=int64))

여기에서 확인

앞에서와 다른 점은 앞에는 컬럼을 버리면서 축소했고 여기서는 다 합쳐서 축소했다.


"""

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




# # 이걸 참고해서 아래 if, elif, else 조건문에 줄일 컬럼을 제한 걸것
# # print(np.unique(y, return_counts=True))
# # # (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))


# # print(y[:10])   # y 값을 앞에서부터 10개만 보여줘라.
# # # [6. 6. 6. 6. 6. 6. 6. 6. 6. 6.]   6이 너무 많네;;; 역시!  갯수가 2198이라서 그런거야 


# ####[ 메인 ]##########################################################################################################

# # y 라벨을 줄일거임 이거는 업무에서만 사용하셈 회사에서는 오더한테 허락받아야하고 캐글에서는 사용불가!##
# # print(y[:20]) 
# # [6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 5. 5. 5. 7. 5. 7. 6. 8. 6. 5.]   # i 안에  6 ~ 5까지 순차적으로 들어감

# for index, value in enumerate(y):  # y를 다 넣음   y 범위 3 ~ 9
#     if value == 9:    # 9.0        5
#         y[index] = 7  
#     elif value == 8:  # 8.0        175
#         y[index] = 7
#     elif value == 7:  # 7.0        880
#         y[index] = 7
#     elif value == 6:  # 6.0        2198
#         y[index] = 6
#     elif value == 5:  # 5.0        1457
#         y[index] = 5
#     elif value == 4:  # 4.0         163
#         y[index] = 4
#     elif value == 3:  # 3.0          20
#         y[index] = 4
#     else:
#         y[index] = 0    
        
# print(np.unique(y, return_counts=True))    # (array([4., 5., 6., 7.]), array([ 183, 1457, 2198, 1060], dtype=int64))


# ########################################################################################################################






# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# x_train, x_test, y_train, y_test = train_test_split(x, y,
#                                                     train_size=0.8,
#                                                     shuffle=True,
#                                                     random_state=123,
#                                                     )

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()

# #3. 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# y_predict = model.predict(x_test)

# score = model.score(x_test, y_test)
# print("model.score : ", score)

# from sklearn.metrics import accuracy_score, f1_score
# print('acc_score : ', accuracy_score(y_test, y_predict))
# print("f1_score(macro) : ", f1_score(y_test, y_predict, average='macro'))
# # print("f1_score(micro) : ", f1_score(y_test, y_predict, average='micro'))




# # ###[ 기본형태 ] ######################

# # model.score :  0.6928571428571428
# # acc_score :  0.6928571428571428
# # f1_score(macro) :  0.39742311190946145

# # ###[ y컬럼 줄인 후 ] ##################

# # model.score :  0.7112244897959183
# # acc_score :  0.7112244897959183
# # f1_score(macro) :  0.6021913708414597

