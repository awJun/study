import pandas as pd
import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import csv


#call data
train_data = pd.read_csv('/Users/jungry/Desktop/git_study/2020_final_project/ratings_train.txt',header=0,delimiter='\t',quoting=3)
test_data = pd.read_csv('/Users/jungry/Desktop/git_study/2020_final_project/data.csv',header=0)

#데이터 전처리 함수
def preprocessing(review,fire_dragon=[]):
    #한글 이외의 것 제거
    review_text=re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","",review)
    #Okt를 사용해 형태소 단위로 쪼개주기
    okt=Okt()
    word_review = okt.morphs(review_text,stem=True)
    #불용어 제거하기
    fire_word = [i for i in word_review if not i in fire_dragon]

    return fire_word  #가볍게 전처리를 한 문자열 반환

fire_dragon = ['의','이','있','하','들','그','되','수','보','않','없','나','사람','아','등','같','오','있','한'] #불용어사전
clean_train_reviews = [] #train data 전처리한거
clean_test_reviews = [] #test data 전처리 한거

#중복 데이터 제거     # drop_duplicates : 중복 제거   /  https://wikidocs.net/154060
train_data.drop_duplicates(subset=['document'],inplace=True)  # inplace : 원본을 변경할지의 여부입니다.
test_data.drop_duplicates(subset=['document'],inplace=True) 

#데이터 전처리
for review in train_data['document']:
    if type(review) == str:
        clean_train_reviews.append(preprocessing(review,fire_dragon=fire_dragon))  #  preprocessing(review,fire_dragon=[]): # fire_dragon 안에 해당되는 불용어 제거
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

with open('clean_test_reviews.csv','w',newline='') as f:  # open(파일이름, w:텍스트 모드로 쓰기)
    writer = csv.writer(f)                                # newline=''은 파일이 '이상한' 것을 방지합니다. 이 기술은 csv 모듈이 모든 플랫폼에서 작동하도록 합니다.
    writer.writerow(clean_test_reviews)          # writer.writerow()명령은 항목을 CSV 파일에 행 단위로 삽입합니다.
    writer.writerow(list(test_data['label']))
                    
                    
                    