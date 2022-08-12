# 결측치 처리  /  결측치 : nan

#1. 행 또는 열 삭제  / 열삭제는 좀 무식한 방법이긴하다고함

#2. 임의의 값  
# 평균 : mean
# 중위 : median
# 0  : fillna
# 앞 : ffill
# 뒤 : bfill
# 특정값 : ....
# 기타등등 : ....

#3. 보간 - interpolate  / 선형회기 방식으로 찾는다. lieanr

#4. 모델 - predict 모델을 돌리고 결측치에 대한 예측값을 뽑는다. 이때 결측치에 대한 y값을 넣어야한다.

# 부스팅꼐열 이상치에 대해 자유롭다 믿거나 말거나 ~ ?? 선생님 ?

import pandas as pd
import numpy as np
from datetime import datetime

dates = ['8/10/2022', '8/11/2022', '8/12/2022','8/13/2022', '8/14/2022']

dates = pd.to_datetime(dates)  # 데이트 타임으로 변경함
print(dates)

print("======================================================")
ts = pd.Series([2, np.nan, np.nan, 8, 10], index=dates)   # Series는 컬럼 한 개짜리라고 생각하면 돼~~  / index = 인덱스로 지정하고 싶은 리스트 넣기
print(ts)

print("======================================================")
ts = ts.interpolate()    # nan을 처리해준다.   interpolate 이거 괜찮은 놈이야 ~ 쓸만해 ~
print(ts)







