"""
[핵심]
ellip = EllipticEnvelope(contamination=.1) 
# 해당 데이터 범위에서 .1(10%)를 이상치로 잡겠다  //  contamination는 이상치

[이상치 보는 법]
[-1  1  1  1  1  1  1  1  1  1  1  1 -1] 
-1은 이상치  / 1은 정상



[에러모음]
ValueError: Expected 2D array, got 1D array instead:   reshape를 안해서 에러남

"""

from sklearn.covariance import EllipticEnvelope
import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
aaa = np.transpose(aaa)

a = aaa[:,[0]]
b = aaa[:,[1]]

ellip = EllipticEnvelope(contamination=.1) # 해당 데이터 범위에서 .1(10%)를 이상치로 잡겠다
ellip.fit(a)
results1 = ellip.predict(a)
print(results1)

ellip.fit(b)
results2 = ellip.predict(b)
print(results2)

# [-1  1  1  1  1  1  1  1  1  1  1  1 -1]
# [ 1  1  1  1  1  1 -1  1  1 -1  1  1  1]