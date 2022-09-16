 #1. 데이터
import numpy as np
from sklearn.metrics import log_loss

x = np.array([1, 2, 3])  # 배열 리스트 1, 2, 3   # 리스트 한 덩어리가 노드에 들어간다.   
y = np.array([1, 2, 3])    
           
              
#2. 모델구성                      
        
from tensorflow.keras.models import Sequential # hiden이 위에서 아래로 내려가는 
# "tensorflow.keras.models" 안에 있는 Seqential를 불러와서 사용하겠다라고 선언. 
#  Sequence 자료형에 속하는 객체는 (문자열, 리스트, 튜플) 
 
from tensorflow.keras.layers import Dense # Dense: 밀도
# https://sevillabk.github.io/Dense/   Dense 설명링크

model = Sequential()  # model = 변수명
model.add(Dense(4, input_dim=1)) #dim 디멘션: 차원 1   # 1은 입력값, 4는 출력값  총 가중치 = 1*4 = 4   
# model.add(...)   층 추가
# 첫번째 인자(4) = 출력 뉴런의 수.
# input_dim(1) = 입력 뉴런의 수. (입력의 차원)
# activation = 활성화 함수.              https://wikidocs.net/32105 관련 링크

model.add(Dense(5))  # 안의 숫자는 노드의 양
model.add(Dense(3))  
model.add(Dense(2))  
model.add(Dense(1))  # 최종 y의 값 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')    
# lose: 오차값을 계산    optimizer: MSE에 최적화를 adam으로 하겠다인거 같음 ;;   
model.fit(x, y, epochs=1000)  # ( ) 괄호 안에는 각각 다 파라미터라고 부름    
# fit: 훈련을 시킬 것 이다 (x, y를 100번)                                                  
# 훈련시킬 x y ㄱ밧을 달라                                                                                                   
# 이 과정을 통해 w 값이 생겼다 이 과정이 lo w      

# model에 가중된 w값이 들어있다.


#4. 평가, 예측
 #평가
loss = model.evaluate(x, y) 
# 갱신된 model의 lose 같을 받아오고 출력
print('loss : ', loss)

 #예측
result = model.predict([4])
# w x 4의 값을 rese 에 넣음
print('4의 예측값 : ', result)
# 3.9999999 or 4.00000001

# loss :  2.2804869104220415e-11
# 4의 예측값 :  [[4.00001]]



# 데이터 전후처리는 훈련전 가장 중요 이 과정으로 정제된 데이터를 추출할 수 있다.

# (하이퍼 파라미터 튜닝 과정)
# 훈련량 레이어 노드  
 
 
   
  
    
   
     



# loss 종류 mae 절대값,  mse 제곱으로

# 스칼라 백터 디멘션에 대해서 정의
# 깃허브 만들고 그 안에 오늘 한 것들 저장해두기

# zkfmakekd@naver.com  반장 이메일
# 이메일로 본인 깃허브 주소 보내기
        
 
