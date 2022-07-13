"""===[ scaler 종류 ]==================================================================================================================

from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
                                ,MaxAbsScaler, RobustScaler

scaler =  MinMaxScaler()
scaler = StandardScaler()
scaler = MaxAbsScaler()
scaler = RobustScaler()

-[ scaler 사용 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

scaler =  MinMaxScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 

 fit 작업: 특성 열의 최소값과 최대값을 찾습니다(이 스케일링은 데이터 프레임 속성/열 각각에 대해
                                      별도로 적용됨을 염두에 두십시오)
                                      
-[ scaler 사용 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

scaler =  MinMaxScaler()

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 1.0000000000000002
# print(np.min(x_test))   # -0.06141956477526944
# print(np.max(x_test))   # 1.1478180091225068

-[ scaler 사용 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# MaxAbsScaler : 절대값이 0~1사이에 매핑되도록 한다. 즉 -1~1 사이로 재조정한다. 
# 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.

# RobustScaler : 아웃라이어의 영향을 최소화한 기법이다. 중앙값(median)과 IQR(interquartile range)을 사용하기 때문에 
# StandardScaler와 비교해보면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있다.

========================================================================================================================
"""