"""=[ pd.read_csv 사용]==============================================================================================

# 해당 경로에 있는 csv파일을 불러옴 

import pandas as pd

train_set = pd.read_csv('./_data/ddarung/train.csv', index_col=0)  
                                                       index_col=0   0번째 인덱스로 지정
                                                       