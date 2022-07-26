from tensorflow.python.keras.models import Model
from keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.python.keras.layers import LSTM, Dense, Embedding, Input

# 1. data
docs = ['너무 재밋어요', '참 최고예요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로예요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밋네요', '민수가 못생기긴 했어요',
        '안결 혼해요'
        ]

# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0]) 

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

'''
{'참': 1, '너무': 2, '재밋어요': 3, '최고예요': 4, '잘': 5, '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번
': 12, '더': 13, '보고': 14, '싶네요': 15, '글쎄요': 16, '별로예요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없 
어요': 22, '재미없다': 23, '재밋네요': 24, '민수가': 25, '못생기긴': 26, '했어요': 27, '안결': 28, '혼해요': 29}
'''

x = token.texts_to_sequences(docs)
print(x)
'''
[[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]
'''
from keras.preprocessing.sequence import pad_sequences 
pad_x = pad_sequences(x, padding='post', maxlen=5) # post도 존재
print(pad_x)
print(pad_x.shape) # (14, 5)

word_size = len(token.word_index)
print(word_size) # 28 : 단어의 종류가 28개

print(np.unique(pad_x))
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]
# 단어사전의 개수는 0부터 시작하기때문에 27 + 1 개이다.

# 원핫인코딩하면 (13, 5) -> (13, 5, 27)

from tensorflow.python.keras.models import Sequential





# 2. model
# model = Sequential()
# model.add(Embedding(input_dim=28, output_dim=77, input_length=5)) # (N, 5, 77) 3차원
# # input_dim : 단어사전의 개수, 라벨의 개수 * 단어사전의 개수보다 많으면 정상적으로 실행된다.
# # output_dim : 임의의 값 * 하이퍼파라미터튜닝의 값으로 어떤값을 줘도 상관없다.
# # input_length : 단어수, 길이 *가급적 max_length에 맞춰줘라
# # model.add(Embedding(27, 77)) # input_dim, output_dim                                                                                     nput_length는 자동으로 고정
# model.add(LSTM(32)) # LSTM : 3차원을 받아들인다.
# model.add(Dense(1, activation='sigmoid'))

# 함수형으로 변환

input1 = Input(shape=(5,)) # input layer (14,5)
# input1 = Input(shape=(None,))
em = Embedding(input_dim=30, output_dim=10, input_length=5)(input1)
em = LSTM(units=32, activation='relu')(em)
output = Dense(1, activation='sigmoid')(em)

model = Model(inputs=input, outputs=output)

model.summary()


'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 77)             2079 
_________________________________________________________________
lstm (LSTM)                  (None, 32)                14080
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 16,192
Trainable params: 16,192
Non-trainable params: 0
'''

# embedding Param : input_dim * output_dim 

#3. compile, fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=32)

#4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print("acc : ", acc)