"""
[핵심]
import matplotlib.pyplot as plt
import math
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])   # 이상치 위치

    print('1사분위: ', quartile_1)                                       
    print('q2: ', q2)
    print('3사분위: ', quartile_3)
    iqr = quartile_3-quartile_1                                          # interquartile range
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    print(upper_bound)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))     # where는 np의 조건문 같음..

def outliers_printer(dataset):                                           # 이상치 위치와 분포도 확인하여 참고하기 위해 그래프를 그려본다.
    plt.figure(figsize=(10,8))
    for i in range(dataset.shape[1]):
        col = dataset[:, i]
        outliers_loc = outliers(col)
        print(i, '열의 이상치의 위치: ', outliers_loc, '\n')
        plt.subplot(math.ceil(dataset.shape[1]/2),2,i+1)
        plt.boxplot(col)
        
    plt.show()

a3 = [  7,  16,  24,  27,  50,  59,  63,  68,  71,  85,  88, 119, 159,
       164, 171, 176, 180, 182, 201, 229, 233, 261, 266, 278, 324, 341,
       374, 386, 409, 480, 485, 541, 542, 634, 642, 683, 686, 726, 787,
       792, 813, 819, 824, 846, 850, 863]

for i in range(len(a3)):   # 위에 a3에 있는 리스트의 길이만큼 반복하겠다.   
    x[a3[i]][3] = 0        # x안에 4번째 열에 이상치가 있기에 앞에 x를 맨뒤 [4]를한다. 
                           # 그 뒤에[a3]인덱스 번호에 해당하는 리스트를 i로 순회돌며 0인 값으로 채운다.



[TIP]
  - 랜덤포레스트 넣고 훈련에서  random_state=1234 값을 잘 설정하는게 값이 가장 잘 나오는 거 같다고 함
 
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

# 1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path+'train.csv')
test_set = pd.read_csv(path+'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
print(train_set['Embarked'].mode())
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

# train_set 불러올 때와 마찬가지로 전처리시켜야 model.predict에 넣어서 y값 구하기가 가능함-----------
print(test_set.isnull().sum())
test_set = test_set.drop(columns='Cabin', axis=1)
test_set['Age'].fillna(test_set['Age'].mean(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].mean(), inplace=True)
print(test_set['Embarked'].mode())
test_set['Embarked'].fillna(test_set['Embarked'].mode()[0], inplace=True)
test_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
test_set = test_set.drop(columns = ['PassengerId','Name','Ticket'],axis=1)
#---------------------------------------------------------------------------------------------------

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
print(x)

x = np.array(x)
y = np.array(y).reshape(-1, 1) # 벡터로 표시되어 있는 y데이터를 행렬로 전환



import matplotlib.pyplot as plt
import math
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])

    print('1사분위: ', quartile_1)
    print('q2: ', q2)
    print('3사분위: ', quartile_3)
    iqr = quartile_3-quartile_1 # interquartile range
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    print(upper_bound)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

def outliers_printer(dataset):
    plt.figure(figsize=(128,128))
    for i in range(dataset.shape[1]):
        col = dataset[:, i]
        outliers_loc = outliers(col)
        print(i, '열의 이상치의 위치: ', outliers_loc, '\n')
        plt.subplot(math.ceil(dataset.shape[1]/2),2,i+1)
        plt.boxplot(col)
        
    plt.show()
    

outliers_printer(x)


a2 = [  7,  11,  15,  16,  33,  54,  78,  94,  96, 116, 119, 152, 164,
       170, 172, 174, 183, 195, 205, 232, 252, 268, 275, 280, 297, 305,
       326, 340, 366, 381, 386, 438, 456, 467, 469, 479, 483, 487, 492,
       493, 530, 545, 555, 570, 587, 625, 626, 630, 642, 644, 647, 659,
       672, 684, 694, 745, 755, 772, 788, 803, 824, 827, 829, 831, 851,
       879]

a3 = [  7,  16,  24,  27,  50,  59,  63,  68,  71,  85,  88, 119, 159,
       164, 171, 176, 180, 182, 201, 229, 233, 261, 266, 278, 324, 341,
       374, 386, 409, 480, 485, 541, 542, 634, 642, 683, 686, 726, 787,
       792, 813, 819, 824, 846, 850, 863]

a4 = [  7,   8,  10,  13,  16,  24,  25,  27,  43,  50,  54,  58,  59,
        63,  65,  68,  71,  78,  86,  88,  93,  97,  98, 102, 118, 119,
       124, 128, 136, 140, 145, 147, 148, 153, 155, 159, 160, 164, 165,
       166, 167, 171, 172, 175, 176, 180, 182, 183, 184, 188, 193, 197,
       201, 205, 229, 233, 237, 247, 248, 251, 254, 255, 259, 261, 262,
       266, 268, 272, 273, 278, 279, 297, 299, 305, 311, 312, 314, 318,
       319, 323, 324, 328, 329, 332, 340, 341, 348, 352, 356, 360, 362,
       374, 377, 381, 386, 390, 394, 407, 409, 416, 417, 419, 423, 424,
       435, 436, 437, 438, 440, 445, 446, 448, 450, 469, 472, 479, 480,
       485, 489, 498, 506, 523, 529, 530, 532, 533, 535, 539, 540, 541,
       542, 548, 549, 550, 558, 567, 580, 581, 585, 587, 593, 595, 600,
       608, 610, 615, 616, 618, 622, 634, 637, 638, 642, 644, 651, 657,
       659, 670, 678, 679, 683, 684, 685, 686, 689, 691, 698, 702, 709,
       720, 736, 742, 745, 746, 750, 751, 754, 755, 763, 774, 779, 783,
       787, 788, 792, 799, 801, 802, 803, 813, 817, 819, 820, 823, 824,
       827, 831, 835, 846, 848, 850, 852, 853, 855, 856, 858, 863, 869,
       871, 879, 880, 885, 888]


a5 = [  1,  27,  31,  34,  52,  61,  62,  72,  88, 102, 118, 120, 124,
       139, 151, 159, 180, 195, 201, 215, 218, 224, 230, 245, 256, 257,
       258, 262, 268, 269, 275, 290, 291, 297, 299, 305, 306, 307, 310,
       311, 318, 319, 324, 325, 332, 334, 336, 337, 341, 366, 369, 373,
       375, 377, 380, 385, 390, 393, 412, 435, 438, 445, 453, 484, 486,
       496, 498, 504, 505, 520, 527, 537, 540, 544, 550, 557, 558, 581,
       585, 587, 591, 609, 627, 641, 645, 655, 659, 660, 665, 679, 681,
       689, 698, 700, 708, 716, 730, 737, 741, 742, 745, 759, 763, 765,
       779, 789, 792, 802, 820, 829, 835, 846, 849, 856, 863, 879]

for i in range(len(a3)):
    x[a3[i]][3] = 0

for i in range(len(a4)):
    x[a4[i]][4] = 0

outliers_printer(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1234)
# lda = LinearDiscriminantAnalysis(n_components=1)
# lda.fit(x_train, y_train) 
# x_train = lda.transform(x_train)
# x_test = lda.transform(x_test)

# 2. 모델
from sklearn.pipeline import make_pipeline
model = make_pipeline(MinMaxScaler(), XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0))

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print(results)

# 0.8324022346368715

# Parch의 이상치를 0으로 채웠을 때
# 0.8435754189944135


#      Pclass  Sex        Age  SibSp  Parch     Fare  Embarked
# 0         3    0  22.000000      1      0   7.2500         0
# 1         1    1  38.000000      1      0  71.2833         1
# 2         3    1  26.000000      0      0   7.9250         0
# 3         1    1  35.000000      1      0  53.1000         0
# 4         3    0  35.000000      0      0   8.0500         0
# ..      ...  ...        ...    ...    ...      ...       ...
# 886       2    0  27.000000      0      0  13.0000         0
# 887       1    1  19.000000      0      0  30.0000         0
# 888       3    1  29.699118      1      2  23.4500         0
# 889       1    0  26.000000      0      0  30.0000         1
# 890       3    0  32.000000      0      0   7.7500         2