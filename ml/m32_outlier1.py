"""
[핵심]
컬럼 하나의 대상으로 이상치 값 위치 확인
"""

import numpy as np

aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])
# aaa = np.array([1, 2, 3, 4, 5])
# aaa = np.array([-10, 1, 2, 4, 5, 6, 7, 8, 10, 50])
# aaa = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, 
                                               [25, 50, 75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound)|   # np.where :  np 조건문 같음
                    (data_out < lower_bound))
    
outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)   # 이상치 : 이상한 값의 위치

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

# 1사분위 :  2.5
# q2 :  5.5
# 3사분위 :  7.75
# iqr :  5.25
# 이상치의 위치 :  (array([2, 8], dtype=int64),)








