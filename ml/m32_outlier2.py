"""
[핵심]
컬럼 여러개의 대상으로 이상치 값 위치 확인


outliers_loc_1 = outliers(aaa[:,0])    # 전체 컬럼중 0번째 위치[:,0
# outliers 위에서 이상치가 빠지만 형태가 바뀌는데 그 형태를 유지하고자
  난값으로 채운다 그걸 채워주는 역할           
"""

import numpy as np

aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]])
aaa = np.transpose(aaa)

# print(aaa.shape)

# print(aaa[:,1])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, 
                                               [25, 50, 75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)     # 중위값  
    lower_bound = quartile_1 - (iqr * 1.5)   # Q1 값
    upper_bound = quartile_3 + (iqr * 1.5)   # q3 값
    return np.where((data_out > upper_bound)|   # np.where :  np 조건문 같음
                    (data_out < lower_bound))
    
outliers_loc_1 = outliers(aaa[:,0])   #  outliers 위에서 이상치가 빠지만 형태가 바뀌는데 그 형태를 유지하고자 난값으로 채운다 그걸 채워주는 역할           전체 컬럼중 0번째 위치[:,0]
outliers_loc_2 = outliers(aaa[:,1])   
print("이상치의 위치 : ", outliers_loc_1)   # 이상치 : 이상한 값의 위치
print("이상치의 위치 : ", outliers_loc_2)   # 이상치 : 이상한 값의 위치

    

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

# 1사분위 :  4.0
# q2 :  7.0
# 3사분위 :  10.0
# iqr :  6.0
# 1사분위 :  200.0
# q2 :  400.0
# 3사분위 :  600.0
# iqr :  400.0
# 이상치의 위치 :  (array([ 0, 12], dtype=int64),)
# 이상치의 위치 :  (array([6], dtype=int64),)

    




