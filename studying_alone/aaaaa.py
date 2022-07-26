# https://happy-obok.tistory.com/41  참고링크

# 1. 필요한 패키지 불러오기 - requests, BeautifulSoup, time, csv
# requests : HTTP 요청을 위해 사용하는 파이썬 라이브러리
# BeautifulSoup : 웹 사이트에서 데이터를 추출하는 웹 스크래핑 라이브러리
# time : 시간 데이터 처리 모듈
# csv : CSV형식의 데이터를 읽고 쓰는 모듈

import requests
from bs4 import BeautifulSoup
import time
import csv


# 2. soup 객체 만들기
#page를 1부터 1씩 증가하며 URL을 다음 페이지로 바꿈 
for page in range(1,500):
    url = f'https://movie.naver.com/movie/point/af/list.naver?&page={page}'
    #get : request로 url의  html문서의 내용 요청
    html = requests.get(url)
    #html을 받아온 문서를 .content로 지정 후 soup객체로 변환
    soup = BeautifulSoup(html.content,'html.parser')
    #find_all : 지정한 태그의 내용을 모두 찾아 리스트로 반환
    reviews = soup.find_all("td",{"class":"title"})


# 3. 데이터 수집하기
#한 페이지의 리뷰 리스트의 리뷰를 하나씩 보면서 데이터 추출
for review in reviews:
    sentence = review.find("a",{"class":"report"}).get("onclick").split("', '")[2]
    #만약 리뷰 내용이 비어있다면 데이터를 사용하지 않음
    if sentence != "":
        movie = review.find("a",{"class":"movie color_b"}).get_text()
        score = review.find("em").get_text()
        review_data.append([movie,sentence,int(score)])
        need_reviews_cnt-= 1



# 4. CSV 파일로 저장
# movie, sentence, score의 열의 데이터 형식으로 추출한 데이터를 저장합니다.
columns_name = ["movie","sentence","score"]
with open ( "samples.csv", "w", newline ="",encoding = 'utf8' ) as f:
    write = csv.writer(f)
    write.writerow(columns_name)
    write.writerows(review_data)


# 5. 다음 페이지를 조회하기 전 시간 두기
# time 라이브러리의 sleep 함수를 사용하면 지정한 시간 동안 프로세스를 일시 정지할 수 있습니다.

time.sleep(0.5)










