from keras.preprocessing.text import Tokenizer

text = '나는 진짜 맛잇는 그 그 밥을 엄청 마구 마구 마구 먹엇다.'

token = Tokenizer()
token.fit_on_texts([text])  # 여기서 인덱스가 생성됨

print(token.word_index)  # 인덱스화가 되었을 때 형태만 확인하는 용도로 찍어봄
# {'마구': 1, '그': 2, '나는': 3, '진짜': 4, '맛잇는': 5, '밥을': 6, '엄청': 7, '먹엇다': 8}
# 순서대로 인덱스 번호를 만들어준다. 단! 반복이 많은 항목이 많을수록 우선순위를 앞으로 보내준다. 
  # 즉 수치화 작업을 해주는 것
  
x = token.texts_to_sequences([text])  # 인덱스화 과정 (이제 사용할 데이터임)
print(x)  # [[3, 4, 5, 2, 2, 6, 7, 1, 1, 1, 8]]   #  2 == '진짜'  /  3 == '맛잇는' /  4 == '밥을' 
 

 
# 3 + 3  = 6  '먹엇다'  ?? 이렇게 되는 것을 방지하기 위해 원핫을 해줘야한다.    즉! 각 숫자를 평등화하는 작업 
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

x = to_categorical(x)
print(x)
print(x.shape)   # (1, 11,  )  <-- LSTM를 사용해야 하는걸 알 수 있음 3차원   즉! 시계열도 사용가능하다.
 
# [[2, 3, 4, 5, 6, 1, 1, 1, 7]]
# [[[0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1.]]]

#####[ 원핫으로 수정해놔라 ~~ ]############## 
import numpy as np
one = OneHotEncoder()
x_new = np.array(sparse=False)
print(x_new.shape)
x_new = one.fit_transform(x_new.reshape(22, 1))
print(x_new)
############################################

















