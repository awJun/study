"""
[여행 상품 신청 여부 예측 경진대회]
https://dacon.io/competitions/official/235959/overview/description

이거는 참고용
https://dacon.io/competitions/official/235959/codeshare/6000?page=1&dtype=recent


id  
연령 : Age
연락처 유형 : TypeofContact
도시 계층 : CityTier
피치 기간 : DurationOfPitch
직업 : Occupation
성별 : Gender
방문 인원 : NumberOfPersonVisiting
후속 조치 횟수 : NumberOfFollowups
제품 피팅 : ProductPitched
선호 속성 스타 : PreferredPropertyStar
결혼 상태 : MaritalStatus
여행 횟수 : NumberOfTrips
여권 : Passport
피치 만족도 점수 : : PitchSatisfactionScore
월별 방문 횟수 : OwnCar
지정소득 : NumberOfChildrenVisiting
월수입 : MonthlyIncome


Company Invited : 회사 초대
Self Enquiry : 셀프 문의

Small Business : 중소 기업
Salaried : 봉급을 받는 사람
Large Business : 대기업

Male : 남자
Female : 여자

Super Deluxe : 더 고급의
Basic : 기본의
Deluxe : 고급의
Standard : 표준

Divorced : 이혼한
Married : 기혼의
Unmarried : 미혼의

Executive : 경영진
Manager : 부장님
VP : 부사장
Senior Manager : 선임 관리자

"""

# 데이콘 문제풀기

from tkinter import Label
import pandas as pd

path = "./_data/dacon_travel/"
train_set = pd.read_csv(path + "train.csv")
test_set = pd.read_csv(path + "test.csv")


train_set.columns = ['아이디', '나이', '연락처_유형', '도시_계층', '피치_기간',
                     '직업', '성별', '방문자_수', '팔로우업_수', '제품_피팅', 
                     '선호된_소유물별', '결혼_상태', '여행_횟수', '여권', 
                     '피치_만족도_점수', '차_소유', '어린이_방문_수', '지정',
                     '월간수입', '탈취한_제품']
# print(train_set.columns)

test_set.columns = ['아이디', '나이', '연락처_유형', '도시_계층', '피치_기간',
                     '직업', '성별', '방문자_수', '팔로우업_수', '제품_피팅', 
                     '선호된_소유물별', '결혼_상태', '여행_횟수', '여권', 
                     '피치_만족도_점수', '차_소유', '어린이_방문_수', '지정',
                     '월간수입']
# print(test_set.columns)

# print(train_set.shape)   # (1955, 20)
# print(test_set.shape)    # (2933, 19)

# print(train_set.isnull().sum())
"""
나이                         94     - 평균도전 
연락처_유형                  10     - 컬럼 통채로 제거 예정
피치_기간                   102    - 평균도전
팔로우업_수                  13     - 평균도전
선호된_소유물별              10     - 평균도전
여행_횟수                    57     - 평균도전
어린이_방문_수               27     - 평균도전
월간수입                    100     - 평균도전
"""
# print(test_set.isnull().sum())
"""
나이                         132     - 평균도전 
연락처_유형                   15     - 드랍 예정
피치_기간                    149     - 평균도전
팔로우업_수                   32     - 평균도전
선호된_소유물별               16     - 평균도전
여행_횟수                     83     - 평균도전
어린이_방문_수                39     - 평균도전
월간수입                     133     - 평균도전
"""

#####[ 결측치 처리 ]###########################################################################

나이_전처리 = train_set['나이'].mean()
# print(나이_전처리)   # 37.46211714132187
train_set['나이'] = train_set['나이'].fillna(나이_전처리)
# print(train_set['나이'].isnull().sum())  # 0

피치_기간_전처리 = train_set['피치_기간'].mean()
# print(피치_기간_전처리)   # 15.524015110631408
train_set['피치_기간'] = train_set['피치_기간'].fillna(피치_기간_전처리)
# print(train_set['피치_기간'].isnull().sum())  # 0

팔로우업_수_전처리 = train_set['팔로우업_수'].mean()
# print(팔로우업_수_전처리)   # 3.718331616889804
train_set['팔로우업_수'] = train_set['팔로우업_수'].fillna(팔로우업_수_전처리)
# print(train_set['팔로우업_수'].isnull().sum())  # 0

선호된_소유물별_전처리 = train_set['선호된_소유물별'].mean()
# print(선호된_소유물별_전처리)   # 3.568637532133676
train_set['선호된_소유물별'] = train_set['선호된_소유물별'].fillna(선호된_소유물별_전처리)
# print(train_set['선호된_소유물별'].isnull().sum())  # 0

여행_횟수_전처리 = train_set['여행_횟수'].mean()
# print(여행_횟수_전처리)   # 3.255532139093783
train_set['여행_횟수'] = train_set['여행_횟수'].fillna(여행_횟수_전처리)
# print(train_set['여행_횟수'].isnull().sum())  # 0

어린이_방문_수_전처리 = train_set['어린이_방문_수'].mean()
# print(어린이_방문_수_전처리)   # 1.213174273858921
train_set['어린이_방문_수'] = train_set['어린이_방문_수'].fillna(어린이_방문_수_전처리)
# print(train_set['어린이_방문_수'].isnull().sum())  # 0

월간수입_전처리 = train_set['월간수입'].mean()
# print(월간수입_전처리)   # 23624.108894878707
train_set['월간수입'] = train_set['월간수입'].fillna(월간수입_전처리)
# print(train_set['월간수입'].isnull().sum())  # 0



###[ 문자 라벨링 ]#################################################################################
"""
train
직업, 성별, 제품_피팅, 결혼_상태, 지정

test
결혼_상태, 지정, 
"""


# ---[ 라벨링할 문자 갯수 확인 ]-----------------------------------------------------------------------------
# print(train_set['직업'].unique())       # ['Small Business' 'Salaried' 'Large Business' 'Free Lancer']
# print(train_set['성별'].unique())       # ['Male' 'Female' 'Fe Male']
# print(train_set['제품_피팅'].unique())  # ['Basic' 'Deluxe' 'King' 'Standard' 'Super Deluxe']
# print(train_set['결혼_상태'].unique())  # ['Married' 'Single' 'Divorced' 'Unmarried']
# print(train_set['지정'].unique())       # ['Executive' 'Manager' 'VP' 'Senior Manager' 'AVP']

# print(test_set['결혼_상태'].unique())   # ['Married' 'Unmarried' 'Divorced' 'Single']
# print(test_set['지정'].unique())        # ['Manager' 'Executive' 'Senior Manager' 'VP' 'AVP']

# train_set['직업'] = {'Small Business' : 0, 'Salaried' : 1,  'Large Business' : 2, 'Free Lancer' : 3}
# train_set['성별'] = {'Male' : 0, 'Female' : 1, 'Fe Male' : 2}                    
# train_set['제품_피팅'] = {'Basic' : 0, 'Deluxe' : 1, 'King' : 2, 'Standard' : 3,  'Super Deluxe' : 4}

# ---[ 문자 라벨링 처리 ]---------------------------------------------------------------------------------------
   # 컴퓨터가 문자를 인식할 수 있도록 문자를 숫자로 변환하는 작업임
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_set['직업'] = le.fit_transform(train_set['직업'])
train_set['연락처_유형'] = le.fit_transform(train_set['연락처_유형'])
train_set['성별'] = le.fit_transform(train_set['성별'])
train_set['제품_피팅'] = le.fit_transform(train_set['제품_피팅'])
train_set['결혼_상태'] = le.fit_transform(train_set['결혼_상태'])
train_set['지정'] = le.fit_transform(train_set['지정'])
# print(train_set.info())

test_set['직업'] = le.fit_transform(test_set['직업'])
test_set['연락처_유형'] = le.fit_transform(test_set['연락처_유형'])
test_set['성별'] = le.fit_transform(test_set['성별'])
test_set['제품_피팅'] = le.fit_transform(test_set['제품_피팅'])
# print(test_set.info())

import numpy as np
x = train_set.drop('탈취한_제품', axis=1)
y = train_set['탈취한_제품']
x = np.array(x)
y = np.array(y)
y = y.reshape(-1, 1)
test_set = np.array(test_set)
print(x.shape, y.shape) # (1955, 19) (1955, 1)

###[ 이상치 확인 ]

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
    plt.figure(figsize=(10,8))
    for i in range(dataset.shape[1]):
        col = dataset[:, i]
        outliers_loc = outliers(col)
        print(i, '열의 이상치의 위치: ', outliers_loc, '\n')
        plt.subplot(math.ceil(dataset.shape[1]/2),2,i+1)
        plt.boxplot(col)
        
    plt.show()

outliers_printer(x)


parameters = {
            'n_estimators':[100,200,300,400,500],
            'learning_rate':[0.1,0.2,0.3,0.5,1,0.01,0.001],
            'max_depth':[None,2,3,4,5,6,7,8,9,10],
            'gamma':[0,1,2,3,4,5,7,10,100],
            'min_child_weight':[0,0.1,0.001,0.5,1,5,10,100],
            'subsample':[0,0.1,0.2,0.3,0.5,0.7,1],
            'reg_alpha':[0,0.1,0.01,0.001,1,2,10],
            'reg_lambda':[0,0.1,0.01,0.001,1,2,10],
              } 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, shuffle=True)
# print(np.unique(y_train, return_counts=True))


# 2. 모델
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
import time

xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
model = make_pipeline(MinMaxScaler(), HalvingRandomSearchCV(xgb, parameters, cv=5, n_jobs=-1, verbose=2))

# 3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과: ', results)
print('걸린 시간: ', end-start)

# 결과:  0.8516624040920716
# 걸린 시간:  160.53282070159912

# 5. 제출 준비
y_submit = model.predict(test_set)
# submission['ProdTaken'] = y_submit

# submission.to_csv(filepath + 'submission.csv', index = True)   # csv 파일을 만드는 방법
