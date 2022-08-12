"""
[에러모음]
ValueError: Expected 2D array, got 1D array instead:   reshape를 안해서 에러남

"""

import numpy as np

aaa = np.array([[-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50],
                [100, 200, -30, 400, 500, 600, -70000, 800, 900, 1000, 210, 420, 350]])
aaa = np.transpose(aaa)

# print(aaa[:,1]) # 1번째 인덱스 컬럼 전체

aaa_1 = aaa[:,0]
aaa_2 = aaa[:,1]

aaa_1 = aaa_1.reshape(-1, 1)
aaa_2 = aaa_2.reshape(-1, 1)

# print(aaa_1)  #2개의 컬럼을 각각 분리
# print(aaa_2)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.2)    # contamination는 이상치

outliers.fit(aaa_1)
outliers.fit(aaa_2)
results_1 = outliers.predict(aaa_1)
# results_2 = outliers.predict(aaa_2)

print(results_1)
# print(results_2)



# outliers.fit(aaa)
# results = outliers.predict(aaa)
# print(results)

# print(outliers[0])

# def contaminations(outlier):
#     a = outliers.fit(outlier)
#     print(a)
#     return
    
    
# results_1 = contaminations(aaa_1)
# # results_2 = contaminations(aaa_2[:,1])

# print(results_1)
# # # print(results_2)
    




# contamination=.3
# [-1 -1  1  1  1  1  1  1  1  1  1 -1 -1]

# contamination=.2
# [-1 -1  1  1  1  1  1  1  1  1  1 -1 -1]

# contamination=.1
# [-1  1  1  1  1  1  1  1  1  1  1  1 -1]












