"""
[핵심]
하이퍼파라미터를 찾는데 알고리즘으로 찾아서 그리드서치랑 다르게 시간이 매우 빠르다.
이 알고리즘의 신빙성은 알고리즘들 돌려서 확인을하는 과정을 거치면 된다.

model = LGBMRegressor(**params) 모델을 사용해라 **params 의미는 위에서 선언한 파라미터를 딕셔너리 형태로 다 받아오겠다라는 뜻임


 # #의 뜻 : 여러개의 인자를 받겠다.          # 사용방법 : *변수이름    / 받아들이고 싶은 만큼 받아들인다.
    # **의 뜻 : 키워드받겟다(딕셔너리 형태)     # 사용방법 : **변수이름   / 딕셔너리 형태로 받아들인다.



BayesianOptimization : 자동으로 최적의 파라미터를 찾아준다.   / 이게 상당히 잘맞춰서 상당히 좋다! 자세히 알고싶으면 통계수학으로 봐봐 ㅋ ;;
                                                                이거 선생님이랑 실무자도 제대로 설명 못한다고함  ㅋㅋ 그냥 쓰자 ㅋㅋㅋ


함수안에 딕셔너리 형태로 넣어주고 파라미터를 사용하면 된다 


[파라미터 생성]
param_bounds = {"x1" : (-1, 5),
                "x2" : (0, 4)}

def y_function(x1, x2):
    return -x1 **2 - (x2 - 2) **2 + 10    # **은 제곱

from bayes_opt import BayesianOptimization    # pip install bayesian_optimization

optimizer = BayesianOptimization(f=y_function,           # f라는 함수에는 우리가 찾고자하는 파라미터를 넣는다.
                                 pbounds=param_bounds,   # 그 함수에 들어갈 파라미터를 딕셔너리 형태로 넣는다.
                                 random_state=1234
                                 )

이걸하면 아래의 결과가 출력된다.
optimizer.maximize(init_points=2,   # maximize: 최대치를 찾아준다.   / init_points : 초기치 설정
                   n_iter=20        # 나는 20번 돌릴 것이다.
                   )  

(출력결과)
# |   iter    |  target   |    x1     |    x2     |
# -------------------------------------------------
# |  1        |  9.739    |  0.1491   |  2.488    |
# |  2        |  6.052    |  1.626    |  3.141    |
# |  3        |  6.528    |  1.775    |  1.435    |
# |  4        |  9.741    |  0.1599   |  2.483    |
# |  5        |  9.445    | -0.4807   |  1.431    |
# |  6        |  5.0      | -1.0      |  0.0      |
# |  7        |  5.0      | -1.0      |  4.0      |
# |  8        |  8.979    | -1.0      |  2.144    |
# |  9        | -19.0     |  5.0      |  0.0      |
# |  10       | -19.0     |  5.0      |  4.0      |
# |  11       |  5.355    |  0.8034   |  0.0      |
# |  12       |  9.63     |  0.5211   |  1.686    |
# |  13       |  9.98     | -0.1395   |  1.992    |
# |  14       |  9.923    |  0.2629   |  2.086    |
# |  15       |  9.966    |  0.02993  |  1.817    |
# |  16       |  9.996    |  0.01324  |  2.062    |
# |  17       |  10.0     |  0.01454  |  1.991    |
# |  18       |  10.0     |  0.007833 |  1.99     |
# |  19       |  10.0     |  0.002898 |  1.987    |
# |  20       |  9.996    | -0.01311  |  2.064    |
# |  21       |  9.998    | -0.02176  |  1.963    |
# |  22       |  9.999    |  0.01782  |  1.985    |
# =================================================


# 이 부분을 보면 최대값이 x1이 0일때 x2가 2일때 가장 좋다라고 추측할 수 있다.  이것을 보고 파라미터 범위를 좁혀가면서 수정하는 과정이 파라미터 튜닝이다.
# |  17       |  10.0     |  0.01454  |  1.991    |                                    통상적으로 범위를 좁힐때 범위+- 2 ~ 3 정도로 잡는다.
# |  18       |  10.0     |  0.007833 |  1.99   


(이것을 찍어봐도 알 수 있다.)
# print(optimizer.max)    # max : 최대값이 나오는 값을 뽑아준다 optimizer라는 변수는 파라미터 관련 변수를 위에서 설정했으므로 
                                  최대값이 나오는 파라미터를 뽑아준다.
# {'target': 9.999835918969607, 'params': {'x1': 0.00783279093916099, 'x2': 1.9898644972252864}}
여기서 target의 값이 r2_score 또는 accuracy를 의미한다.


위에와 동일한 결과를 볼 수 있다.


"""


from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier  
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings("ignore")   # 잔소리 제거

#1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,   # 디폴트가 True인데 걍 써봄 ㅋ
                                                    random_state=1234,
                                                    stratify=y
                                                    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# bayseian_params = {
#     'colsample_bytree' : (0.5, 1),
#     'max_depth' : (6,16),
#     'min_child_weight' : (1, 50),
#     'reg_alpha' : (0.01, 50),
#     'reg_lambda' : (0.001, 1),
#     'subsample' : (0.5, 1)
# }

bayseian_params = {
    'colsample_bytree' : (0.5, 0.7),
    'max_depth' : (7,10),
    'min_child_weight' : (1, 1.5),
    'reg_alpha' : (16, 20),
    'reg_lambda' : (0.6, 1),
    'subsample' : (0.4, 0.7)
}

def lgb_function(max_depth, min_child_weight,subsample, colsample_bytree, reg_lambda, reg_alpha):
    params ={
        'n_estimators' : 500, 'learning_rate' : 0.02,
        'max_depth' : int(round(max_depth)),                    # 정수만
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1),0),                  # 0~1 사이값만
        'colsample_bytree' : max(min(colsample_bytree,1),0),
        'reg_lambda' : max(reg_lambda,0),                       # 양수만
        'reg_alpha' : max(reg_alpha,0),
    }
    
    # *의 뜻 : 여러개의 인자를 받겠다.          # 사용방법 : *변수이름    / 받아들이고 싶은 만큼 받아들인다.
    # **의 뜻 : 키워드받겟다(딕셔너리 형태)     # 사용방법 : **변수이름   / 딕셔너리 형태로 받아들인다.
    model = XGBClassifier(**params)   # 해당안에 파라미터를 다 받아준다.

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)

    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)   # target으로 출력됨
    
    return results
# 어떤 최대값을 넣었을 때 어떤 파라미터가 나오는지 확인해야함.

lgb_bo = BayesianOptimization(f=lgb_function,
                              pbounds = bayseian_params,
                              random_state=123)

lgb_bo.maximize(init_points=2, n_iter=100)
print(lgb_bo.max)


model = XGBClassifier(n_estimators = 500, learning_rate= 0.02, colsample_bytree =max(min(0.6392938371195723,1),0) , max_depth=int(round(8.716836009702277)), min_child_weight =int(round(1.226851453564203)),
                      reg_alpha= max(18.269721536243367,0), reg_lambda=max(0.7197495008157775,0), subsample=max(min(0.5846212920248922,1),0))

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(score)


# 결과
# {'target': 0.9736842105263158, 'params': {'colsample_bytree': 0.6392938371195723, 'max_depth': 7.8584180048511385,
#   'min_child_weight': 1.1134257267821015, 'reg_alpha': 18.205259076331565, 'reg_lambda': 0.8877875879142252, 
#   'subsample': 0.5269319380373383}}

# 0.9736842105263158

# 성능 동일함