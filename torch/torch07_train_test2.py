import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# [검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법 찾아라

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    train_size = 0.7,  # (둘 중에 하나만 사용하면 된다.)
                                                    test_size=0.3, # (데이터 비율)
                                                    shuffle=True,  # 셔플 데이터를 섞겟다(true면 섞이고 false면 안섞음)
                                                    random_state=12345678) # 랜덤난수 표중에서 66번의 난수값을 사용해라


x_predict = np.array([11,12,13])       

# 시작!!!



