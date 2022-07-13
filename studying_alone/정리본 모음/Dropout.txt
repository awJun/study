"""=[ Dropout 사용 ]==============================================================================================

from tensorflow.python.keras.layers import Dense, Dropout #(데이터의 노드를 중간중간 날려줌)
                                                 # 데이터가 많을수록 성능 좋음.
                    
model = Sequential()
model.add(Dense(100, input_dim=13))
model.add(Dropout(0.3))      # 30%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))       # 20%만큼 노드를 지워버리겠다 (이빨을 빼버리겠다.)
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 노드를 중간중간 날려도 평가, 예측에서는 전체 노드가 다 적용된다.                    
                                                  
========================================================================================================================
"""   