from keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()

token.fit_on_texts([text]) # text에 있는 문자를 읽고 index를 부여해 수치화한다.

print(token.word_index)
#{'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
x = token.texts_to_sequences([text])
print(x)

#[[3, 4, 2, 2, 5, 6, 7, 1, 1, 1, 8]] (10,1)
##############자연어 처리는 시계열 분석이 기본적이다!
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# x = to_categorical(x)
# print(x,x.shape) #(1, 11, 9)  # 카테고리컬은 시작할 때 0아 아닐경우 0부터 만들고 채우기 
# 때문에 8개에서 9개로 1개가 늘어남 그렇기 때문에 지금은 카테고리컬이 원핫인코딩을해서 필요없는 행이
# 늘어나는 것을 방지한 것이다
from sklearn.preprocessing import OneHotEncoder
import numpy as np
x = np.array(x).reshape(-1,1)
ohe = OneHotEncoder()
x = ohe.fit_transform(x).toarray()
print(x,x.shape) #(11, 8)
# [[0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1.]]









