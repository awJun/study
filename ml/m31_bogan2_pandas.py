import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [2, 4, 6, 8, 10],
                     [np.nan, 4, np.nan, 8, np.nan]])

# print(data)
data = data.transpose()   # 행과 열 위치 변환
# print(data)
data.columns = ["x1", "x2", "x3", "x4"]
# print(data.shape)   # (4, 5)
print(data)


# 결측치 확인
# print(data.isnull())
# print(data.isnull().sum())
print(data.info())

#1. 결측치 삭제
print("================[ 결측치 삭제 ]=========================")
print(data.dropna())   # 디폴트는 행 
print(data.dropna(axis=0))    
print(data.dropna(axis=1)) 

#2-1. 특정값 - 평균
print("================[ 결측치 처리 mean() ]=========================")
means = data.mean()   # 컬럼별 평균
print("평균 : ", means)  
data2 = data.fillna(means)
print(data2)

#2-2. 특정값 - 중위값
print("================[ 결측치 처리 median() ]=========================")
medians = data.median()   # 컬럼별 평균
print("중위값 : ", medians)  
data2 = data.fillna(medians)
print(data2)

#2-3. 특정값 - ffill, bfill   /  ffill앞껄로 채우겠다.  bfill뒤에껄로 채우겠다.
print("================[ 결측치 처리 ffill, bfill ]=========================")
data4 = data.fillna(method='ffill')  
print(data4)

data5 = data.fillna(method='bfill')  
print(data5)

#2-4. 특정값 - 임의값으로 채우기
print("=================[결측치 - 임의의값으로 채우기]===================")
# data6 = data.fillna(77777)  
data6 = data.fillna(value = 77777)  # 위 아래 같은거임
print(data6)


############################  특정컬럼만 !!!  #####################################

means = data['x1'].mean()
print(means)    # 6.5
data['x1'] = data['x1'].fillna(means)
print(data['x1'])

meds = data['x2'].median()
print(meds)     # 4.0
data['x2'] = data['x2'].fillna(meds)
print(data['x2'])

data = data['x4'].fillna(77777)   # 임의의 숫자 777을 넣음
print(data)



