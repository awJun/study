from keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2   # 단어사전 갯수 10000개
)

# print(x_train)
# print(x_train.shape, y_train.shape)  # 쉐이브는 나오지만 각 데이터마나의 길이는 알 수 없음 / 데이터 값마다 크기 차이가 심한게 원인같음
# print(y_train)
# print(y_train.shape, y_train.shape)
print(np.unique(y_train, return_counts=True))   # 46개의 뉴스카테고리 즉! 카테고리컬 사용해야함 ㅋ.
print(len(np.unique(y_train)))   # 46 unique의 갯수


# # print(type(x_train), type[y_train])  # <class 'numpy.ndarray'> type[array([ 3,  4,  3, ..., 25,  3, 25], dtype=int64)]
# # print(type(x_train[0]))              # <class 'list'>
# # # print(x_train[0].shape)  # AttributeError: 'list' object has no attribute 'shape' 에러 ; 

# # # 최대와 최소길이를 알기위해
# # print(len(x_train[0]))               # 87  리스트타입이기 대문에 shape로 알 수 없다.
# # print(len(x_train[1]))               # 56

# print(len(max(x_train)))

# "뉴스기사의 최대길이 : ", max(len(i) for i in x_train)    
# "뉴스기사의 최대길이 : ", sum(map(len, x_train)) / len(x_train)
# # len(i)를 사용해서 i에는 리스트의 길이값이 저장된다. max가 붙어서 최대길이가 저장된다

# # 전처리
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils.np_utils import to_categorical

# x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')  # 앞에서부터 0으로 채우고 100개까지
#                         # (8982, )   --> (8982, 100)
# x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')    # 앞에서부터 0으로 채우고 100개까지
                        
# y_train = to_categorical(y_train)   # 0부터 시작하므로 to_categorical을 사용해도 무관함
# y_test = to_categorical(y_test)

# print(x_train.shape, y_train.shape)    # (8982, 100) (8982, 46)
# print(x_test.shape, y_test.shape)      # (2246, 100) (2246, 46)

#2. 모델 구성
# 시작!


























