import time
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs

from tqdm import trange

base_url = "https://movie.naver.com/movie/bi/mi/point.naver?code=81888"
response = requests.get(base_url)

soup = bs(response.text, 'html.parser')  # response.text로 텍스트를 받고 그걸 html.parser로 파싱 작업을함

content = soup.select("body > div > div > div.score_result > ul > li:nth-child(1)") # copy selector

print(content)










