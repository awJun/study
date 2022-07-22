from keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 매우 맛잇는  밥을 엄청 마구 마구 마구 먹엇다.'
text2 = '나는 지구용사 이재근이다. 멋있다 또 얘기해봐'

token = Tokenizer()
token.fit_on_texts([text, text2])  # 여기서 인덱스가 생성됨

print(token.word_index)  # {'마구': 1, '나는': 2, '매우': 3, '진짜': 4, '맛잇는': 5, '밥을': 6, '엄청': 7, '먹엇다': 8, '지구용사': 9, '이재근이다': 10, '멋있다': 11, '또': 12, '얘기해봐': 13}
# 순서대로 인덱스 번호를 만들어준다. 단! 반복이 많은 항목이 많을수록 우선순위를 앞으로 보내준다. 
  # 즉 수치화 작업을 해주는 것
  
x = token.texts_to_sequences([text, text2])
print(x)      # [[2, 4, 3, 3, 5, 6, 7, 1, 1, 1, 8], [2, 9, 10, 11, 12, 13]]   #  2 == '진짜'  /  3 == '맛잇는' /  4 == '밥을' 
 

 
# 3 + 3  = 6  '먹엇다'  ?? 이렇게 되는 것을 방지하기 위해 원핫을 해줘야한다.    즉! 각 숫자를 평등화하는 작업 
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

#to_categorical을 사용하려면 이 과정을 해야함 / 리스트가 2개이므로 동시에 불가능하다고 에러 발생시킴
x_new = x[0] + x[1]    #  [text, text2]를 합친다. 라는 뜻
print(x_new)



# x_new = to_categorical(x_new)
# print(x_new)
# print(x_new.shape)     # (17, 14)  두 개를 합쳐서 이런 형태로 나옴
# # [[2, 3, 4, 5, 6, 1, 1, 1, 7]]
# # [[[0. 0. 1. 0. 0. 0. 0. 0.]
# #   [0. 0. 0. 1. 0. 0. 0. 0.]
# #   [0. 0. 0. 0. 1. 0. 0. 0.]
# #   [0. 0. 0. 0. 0. 1. 0. 0.]
# #   [0. 0. 0. 0. 0. 0. 1. 0.]
# #   [0. 1. 0. 0. 0. 0. 0. 0.]
# #   [0. 1. 0. 0. 0. 0. 0. 0.]
# #   [0. 1. 0. 0. 0. 0. 0. 0.]
# #   [0. 0. 0. 0. 0. 0. 0. 1.]]]

# #####[ 원핫으로 수정해놔라 ~~ ]############## 
# import numpy as np
# one = OneHotEncoder()
# x_new = np.array(sparse=False)
# print(x_new.shape)
# x_new = one.fit_transform(x_new.reshape(22, 1))
# print(x_new)
# ###########################################

















