# 영화 리뷰 개인프로젝트
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

path = './review_data/'
train_set = pd.read_csv(path + '영화 리뷰 train.csv', sep='\t')
# print(train_set)
# [8060 rows x 3 columns]  60개 넘김 (목표달성)

test_set = pd.read_csv(path + '영화 리뷰 test.csv', sep='\t')
# print(test_set)
# [3236 rows x 3 columns]

# print(train_set.isnull().sum())  # 결측치는 없는 것으로 판명
# print(test_set.isnull().sum())  # 결측치는 없는 것으로 판명


#--[ x, y 분리 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
x = train_set['document'] 
# print(x.shape) # (8060,)

y = train_set['label'] 
# print(y.shape) # (8060,)

# print(x)
# print(y)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#---[ 데이터 토큰화 과정 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 긍정 / 부정 값의 분포 확인 (그래프 확인 이미지 캡쳐 완료)
import matplotlib.pyplot as plt
y.value_counts().plot(kind = 'bar', color='blue')   # 라벨의 긍정 부정 분포도 확인
# plt.show()

#--[ 문자를 나눈다. => 토큰화 ]--------
x = str(x).split()
print(len(x))
print(x[:20])

#--[ 한글과 공백을 제외하고 모두 제거 (정규 표현식 수행) ]------------
x = str(x).replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", " ")   # replace는 문자열을 변경하는 함수 https://ooyoung.tistory.com/77
# print(x[:100])   # [:30] 인덱스 번호 0~29까지 출력하겠다

import re   #  
no_num = re.compile('[^0-9]')
# print(no_num)
x = ("".join(no_num.findall(x)))
# print(x[:500]) 

#--[ 불용어 제거 ]-----------------------------------------------
# import nltk
# nltk.download('punkt')   # 다운로드 완료했으므로 주석 처리함.


# print(x)
# print("=======================================================")

# print(x)
# print("==================")
stopwords = ['이', '있', '하', '것', '들', '그', '되', '수', '이', '보', '않', '없', '나',
             '사람', '주', '아니', '등', '같', '우리', '때', '년', '가', '한', '지', '대하',
             '오', '말', '일', '그렇', '위하', '이네요', '나네요', '이네요']

x = x.replace('stopwords', ' ')     # stopwords안에 있는 항목을 다 빈공간으로 처리하겠다는 뜻
# print(x)

# # hi = "Hello, World!"
# # table = str.maketrans('HWd', '123')
# # hi.translate(table)

# # import numpy as np
# # x = np.array(x)

# # print(x.shape)




from keras.preprocessing.text import Tokenizer
token = Tokenizer(oov_token="<OOV>")
token.fit_on_texts([x]) # text에 있는 문자를 읽고 index를 부여해 수치화한다.
print(token.word_index)
#{'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
# x = token.texts_to_sequences([x])
# print(x)

# from sklearn.preprocessing import OneHotEncoder
# import numpy as np
# x = np.array(x).reshape(-1,1)
# ohe = OneHotEncoder()
# x = ohe.fit_transform(x).toarray()

# from keras.preprocessing.sequence import pad_sequences
# pad_x =pad_sequences(x,padding='pre',maxlen=5) #통상 같은 크기를 맞추기 위해서는 앞에서부터 패딩한다. maxlen 최대 글자 수의 제한 
# print(pad_x)
# print(pad_x.shape) #(14, 5)
# word_size = len(token.word_index)
# print("wored_size :",word_size) #단어 사전의 갯수 : 30

# print(np.unique(pad_x,return_counts=True))

# print(pad_x.shape)
# print(y.shape)
 

#--[ 형태소 분리 Okt]------------------------------------------------
# from konlpy.tag import Okt

# okt = Okt()

# print(okt.morphs(u'x'))




# for i, document in enumerate(x):   #enumerate 는 열거하다라는 단어이다. 파이썬에서는 List , Tuple ,
#             # String 등 여러가지 자료형을 입력받으면 인덱스 값을 포함하는 enumerate 객체를 돌려준다.
#     clean_words = [] 
#     for word in nltk.tokenize.word_tokenize(document): 
#         if word not in stopwords: #불용어 제거
#             clean_words.append(word)  
#     print(clean_words) #['스토리', '진짜', '노잼']     
#     x[i] = ' '.join(clean_words) 
# print(x)    
#-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

# def review_to_words( raw_review ):
#     # 5. Stopwords 불용어 제거
#     meaningful_words = [w for w in words if not w in stops]
#     # 6. 어간추출
#     stemming_words = [stemmer.stem(w) for w in meaningful_words]
#     # 7. 공백으로 구분된 문자열로 결합하여 결과를 반환
#     return( ' '.join(stemming_words) )







# #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# #--[ x 데이터 자연어 처리 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# from keras.preprocessing.text import Tokenizer
# import numpy as np       

# token = Tokenizer(oov_token="<OOV>")
# token.fit_on_texts(x)

# x = token.texts_to_sequences(x)

# # print(x)


#2. 모델 구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,Dense,Embedding
model = Sequential()
model.add(Embedding(input_dim=31,output_dim=10,input_length=5)) 
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
model.summary() 


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x, y, epochs=20,batch_size=16)

# #4. 평가, 예측
# acc = model.evaluate(x, y)[1]
# print('acc :',acc)






