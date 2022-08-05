import requests
from bs4 import BeautifulSoup
import time
import csv

#목표량 정하기
#리스트에 데이터담기
need_reviews_cnt = 1000 
reviews = []
review_data=[]

#page를 1부터 1씩 증가하며 URL을 다음 페이지로 바꿈 
for page in range(1,2):
    url = f'https://movie.naver.com/movie/point/af/list.naver?&page={page}'
    #get : request로 url의  html문서의 내용 요청
    html = requests.get(url)
    #html을 받아온 문서를 .content로 지정 후 soup객체로 변환
    soup = BeautifulSoup(html.content,'html.parser')
    #find_all : 지정한 태그의 내용을 모두 찾아 리스트로 반환
    reviews = soup.find_all("td",{"class":"title"})
    
    #한 페이지의 리뷰 리스트의 리뷰를 하나씩 보면서 데이터 추출
    for review in reviews:
        sentence = review.find("a",{"class":"report"}).get("onclick").split("', '")[2]
        #만약 리뷰 내용이 비어있다면 데이터를 사용하지 않음
        if sentence != "":
            movie = review.find("a",{"class":"movie color_b"}).get_text()
            score = review.find("em").get_text()
            review_data.append([movie,sentence,int(score)])
            need_reviews_cnt-= 1
    #현재까지 수집된 리뷰가 목표 수집 리뷰보다 많아진 경우 크롤링 중지        
    if need_reviews_cnt < 0:                                         
        break
    #다음 페이지를 조회하기 전 0.5초 시간 차를 두기
    time.sleep(0.5)
     


columns_name = ["movie","sentence","score"]
with open ( "naverMovieReview.csv", "w", newline ="",encoding = 'utf8' ) as f:
    write = csv.writer(f)
    write.writerow(columns_name)
    write.writerows(review_data)