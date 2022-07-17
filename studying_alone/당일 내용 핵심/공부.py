# https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf 판다스 시트자료
# https://pandas.pydata.org/docs/user_guide/10min.html 판다스 10분 종 합문서


# [해당 컬럼의 고유값을 알려준다.]


import pandas as pd

#--[참고로 알아두면 좋음]- - - - - - - - - - - - - - - - - - - - - - - - -  

df = pd.DataFrame([])   # 비어있는(아무것도 안들어있는)데이터 프레임을 만듬
# print(df)   # 데이터 안에 아무것도 없음을 확인함
# Empty DataFrame
# Columns: []
# Index: []

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import numpy as np
# df = pd.DataFrame({'A' : range(1, 11), 'B' : np.random.randn(10)})   #  np.random.randn을 이용해서 랜덤의 수를 만들어서 넣음
# print(df)
#     A         B
# 0   1 -0.336371
# 1   2 -0.221677
# 2   3 -0.494146
# 3   4 -1.019765
# 4   5  1.456598
# 5   6  1.233492
# 6   7 -1.926307
# 7   8  0.805534
# 8   9  0.650800
# 9  10  0.077304















































