# [SimpleRNN] units : 10 -> 10* (1 + 1 +10) = 120

# [LSTM] units : 10 -> 4 * 10 * (1 + 1 + 10) = 480
                    #    4 * 20 * (1 + 1 + 20) = 1760
# [결론] LSTM = simpleRNN * 4
# 숫자4의 의미는 cell state, input gate, output gate, forget gate
# 한마디로 LSTM이 SimpleRNN보다  연산량이 많다 성능도 좋다 하지만 안좋은 경우도 있다 둘 다 사용해보고 알아서 튜닝할 것 ㅋ

# [GPU] units : 10 -> 3  10 * (1 + 1 + 10) = 360

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# model.add(SimpleRNN(units=100 ,input_length=3, input_dim=1))
#                     SimpleRNN -->  LSTM로 바꾸는 방법 그냥 앞에꺼만 바꾸면 됨 ㅋ
# model.add(LSTM(units=100 ,input_length=3, input_dim=1))




# LSTM가 SimpleRNN보다 성능이 좋다 (그만큼 연산량이 많다.)
# LSTM 설명 링크  https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=magnking&logNo=221311273459

#=======================================================================================================================

import numpy as np

#1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
             [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
             [9, 10, 11], [10, 11, 12],
             [20, 30, 40], [30, 40, 50], [40, 50, 60]]
             )
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])        # 80이라는 숫자가 나오도록 예측하자 ~      50, 60, 70을 예측에 넣으면 될거같음

#2. 모델구성
from tensorflow.python.keras.models import Sequential              
from tensorflow.python.keras.layers import LSTM, Dense              

##########################################################################
model = Sequential()                                            #  return_sequences=True 해당 레이어에서 던져주는 것들을 리쉐입을 안해도 3차원으로 던져줘서 그대로 넘어간다.
model.add(LSTM(10, return_sequences=True, input_shape=(3, 1)))  #  (N, 3, 1)  -->  (N, 3, 10) 필터부분만 변경됨
model.add(LSTM(5, return_sequences=False))      # True면 차원이 늘어난다 / False가 디폴트값 차원이 안들어남 즉! 사용하나 마다 ~~ 위에서 3차원으로 받아야할 때  False하면 바로 에러남
model.add(Dense(1))
model.summary()
############################################################################

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# lstm (LSTM)                  (None, 3, 10)             480       
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 5)                 320       
# _________________________________________________________________
# dense (Dense)                (None, 1)                 6
# =================================================================
# Total params: 806
# Trainable params: 806
# Non-trainable params: 0
# _________________________________________________________________


# LSTM 2개 엮은거 테스트해보기

























