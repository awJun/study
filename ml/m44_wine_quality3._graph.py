import pandas as pd
import warnings
warnings.filterwarnings('ignore') # warnig 출력 안함

path = "D:/study_data/_data/"

datasets = pd.read_csv(path+'winequality-white.csv',index_col=None, header=0, sep=';')

# print(datasets.shape)  # (4898, 12)
# print(datasets.describe())   # pandas에서만 사용가능
# print(datasets.info())   # 결측치 확인

#################### 그래프 그려봐 ##########################################
#1. value_count -> 쓰지말고 해볼것
#2. groupby 써, count 써!!!!

# plt.bar 로 그린다. (quality 컬럼)

# count_data = datasts. groupby count     # 여기서는 판다스의 groupby임
# plt.bar(주저리주저리)
# plt.show()
import matplotlib.pyplot as plt
count_data = datasets.groupby('quality')['quality'].count()
plt.bar(count_data.index, count_data)
plt.show()


