"""
[에러모음]
ValueError: Expected 2D array, got 1D array instead:   reshape를 안해서 에러남

"""

import numpy as np

aaa = np.array([-10, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 50])
aaa = aaa.reshape(-1, 1)
# print(aaa.shape)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)    # contamination는 이상치

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results)





# contamination=.3
# [-1 -1  1  1  1  1  1  1  1  1  1 -1 -1]

# contamination=.2
# [-1 -1  1  1  1  1  1  1  1  1  1 -1 -1]

# contamination=.1
# [-1  1  1  1  1  1  1  1  1  1  1  1 -1]












