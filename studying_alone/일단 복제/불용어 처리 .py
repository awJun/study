import pandas as pd
from tqdm import tqdm
from konlpy.tag import Okt
from sklearn. feature_extraction.text import CountVectorizer

# 자연어 처리
def NLP_DTM():
    # 타이틀 리스트를 부럴와서 title_list 변수에 저장한다.
    t_file_name = open('./review_data/영화 리뷰 train.csv', sep='\t')
    print(t_file_name)


















