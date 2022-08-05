




# 영화 리뷰 개인프로젝트
from re import X
from bs4 import BeautifulSoup
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LSTM   
from tensorflow.python.keras.models import Sequential
import time
from sklearn.model_selection import train_test_split # 함수 가능성이 높음
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 클래스 가능성이 높음
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error



#1. 데이터
import pandas as pd  
from keras.preprocessing.text import Tokenizer
import numpy as np          

path = './review_data/'
# print(train_set.isnull().sum())  # 결측치는 없는 것으로 판명
# print(test_set.isnull().sum())  # 결측치는 없는 것으로 판명

datasets = pd.read_csv(path + 'datasets.csv', sep='\t')

# from bs4 import BeautifulSoup

example1 = BeautifulSoup(datasets['document'][0], 'html5lib')
# print(datasets['document'][0][:700])

# HTML 제거
data = example1.get_text()
# print(data)

# x = data.replace("더빙", " ")   # replace는 문자열을 변경하는 함수 https://ooyoung.tistory.com/77
# print(x)

from soynlp.tokenizer import MaxScoreTokenizer

scores = {'파스': 0.3, '파스타': 0.7, '좋아요': 0.2, '좋아':0.5}
tokenizer = MaxScoreTokenizer(scores=scores)

print(tokenizer.tokenize('난파스타가좋아요'))


# token = Tokenizer()
# token.fit_on_texts(x)
# print(token.word_index)
# # {'참': 1, '너무': 2, '재밌어요': 3, '최고에요': 4, '잘': 5, '만든': 6, '영화에요': 7, 
# #  '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '
# #  싶네요': 15, '글세요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, 
# #  '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '민수가': 25, '못': 26,
# #  '생기기긴': 27, '했어요': 28, '안결': 29, '혼해요': 30}

# x = token.texts_to_sequences(x)
# # print(x) 




# #각 문장마다 길이가 다르기 때문에 가장 큰 값을 기준으로 모자른 값들은 0으로 채운다.
# #값이 너무 클경우 일정양으로 자른다.
# # [[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15],
# #  [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 26, 27, 28], [29, 30]]
# from keras.preprocessing.sequence import pad_sequences
# pad_x =pad_sequences(x,padding='pre',maxlen=5) #통상 같은 크기를 맞추기 위해서는 앞에서부터 패딩한다. maxlen 최대 글자 수의 제한 
# print(pad_x)
# print(pad_x.shape) #(14, 5)
# word_size = len(token.word_index)
# print("wored_size :",word_size) #단어 사전의 갯수 : 30

# print(np.unique(pad_x,return_counts=True))
# # (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
# #        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]), 
# # array([37,  3,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
# #         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],



# # # print(datasets.shape)  # (15797, 3)
# # # #--[ x, y 분리 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# # x = datasets['document'] 
# # # print(x.shape) # (15797,)

# # y = datasets['label'] 
# # # print(y.shape) # (15797,)

# # # print(x)
# # # print(y)
# # # #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# # from keras.preprocessing.text import Tokenizer













