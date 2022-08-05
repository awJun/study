import pandas as pd
import re
from konlpy.tag import Okt
import numpy as np
import csv


#1. 데이터
path = './review_data/'
train_data = pd.read_csv(path + '영화 리뷰 train.csv', sep='\t')
# print(train_set)
# [8060 rows x 3 columns]  60개 넘김 (목표달성)

test_data = pd.read_csv(path + '영화 리뷰 test.csv', sep='\t')
# print(test_set)
# [3236 rows x 3 columns]

# print(train_set.isnull().sum())  # 결측치는 없는 것으로 판명
# print(test_set.isnull().sum())  # 결측치는 없는 것으로 판명

#데이터 전처리 함수
def preprocessing(review,fire_dragon=[]):
    #한글 이외의 것 제거
    review_text=re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","",review)
    #Okt를 사용해 형태소 단위로 쪼개주  기
    okt=Okt()
    word_review = okt.morphs(review_text,stem=True)
    #불용어 제거하기
    fire_word = [i for i in word_review if not i in fire_dragon]

    return fire_word  #가볍게 전처리를 한 문자열 반환

fire_dragon = ['의','이','있','하','들','그','되','수','보','않','없','나','사람','아','등','같','오','있','한'] #불용어사전
clean_train_reviews = [] #train data 전처리한거
clean_test_reviews = [] #test data 전처리 한거

#중복 데이터 제거  # drop_duplicates : 중복요소 제거  /  https://wikidocs.net/154060
train_data.drop_duplicates(subset=['document'],inplace=True)   # inplace : 원본을 변경할지의 여부입니다.
test_data.drop_duplicates(subset=['document'],inplace=True) 

#데이터 전처리    
for review in train_data['document']:
    if type(review) == str:
        clean_train_reviews.append(preprocessing(review,fire_dragon=fire_dragon))
    else:
        clean_train_reviews.append([])

for review in test_data['document']:
    if type(review) == str:
        clean_test_reviews.append(preprocessing(review,fire_dragon=fire_dragon))
    else:
        clean_test_reviews.append([])

#전처리 데이터 파일로 저장 (시간절약)
with open('clean_train_reviews.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(clean_train_reviews)
    writer.writerow(list(train_data['label']))

with open('clean_test_reviews.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(clean_test_reviews)
    writer.writerow(list(test_data['label']))



# #전처리 데이터 파일로 저장 (시간절약)
# import pickle
# with open('clean_train_reviews.csv','w',newline='') as f:
# with open('data.pickle', 'wb') as f:
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)




