"""
[핵심]
DataFrame 형태를 넘파이로 변경해줘야 torch.FloatTensor에서 에러가 발생하지 않는다.

 - 모델의 intput과 output의 형태가 안받으면 아래와 같은 에러 발생함
   RuntimeError: mat1 and mat2 shapes cannot be multiplied (1167x9 and 8x64)


[데이터가 커서 터짐] 안돌아감
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
print("torch : ", torch.__version__, "사용DEVICE : ", DEVICE)

#1. 데이터
import pandas as pd
path = './_data/kaggle_bike/'  

   
     # 결측치는 없는 것으로 판명                                                               
train_set = pd.read_csv(path + 'train.csv', index_col=0)  
  # (10886, 11)                
test_set = pd.read_csv(path + 'test.csv', index_col=0) 
  # (6493, 8)

x = train_set.drop(['casual',  'registered',  'count'], axis=1)  #       axis = 0/1(축값, 0=행, 1=열)                                                                    
   # (10886, 8)
y = train_set['count']
print(type(x))

x = x.to_numpy()
y = y.to_numpy()

# import numpy
# print(numpy.unique(y))  # 회귀

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=100
                                                    )

x_train = torch.FloatTensor(x_train)
# y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_train = torch.FloatTensor(y_train).to(DEVICE)   # int가 길어지면 Long 
x_test = torch.FloatTensor(x_test)
# y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)  # Float가 커지면 doble
y_test = torch.FloatTensor(y_test).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size())   # torch.Size([8708, 8])


#2. 모델
model = nn.Sequential(
    nn.Linear(8, 64),   # 앞에는 컬럼 갯수  
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 16),   # 마지막은 y의 유니크값 갯수 3개  # 클래스 갯 수에 맞춰서 해야한다.
    nn.Linear(16, 1),   # softmax를 안해줘도 된다.
).to(DEVICE)

#3. 컴파일, 훈련
# critenrion = nn.BCELoss()    # # BCE : 바이너리크로스엔트로피   / 이진분류
# critenrion = nn.CrossEntropyLoss()    # # BCE : 바이너리크로스엔트로피   / 이진분류   / CrossEntropyLoss : spars_crossEntropy  
critenrion = nn.MSELoss() # = loss

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train):
    # model. train()
    optimizer.zero_grad()     # 역전파 이후 개인된 것으로 돌아갈 때 메모리에 잡것들이 남아있다고한다. 이것들을 제거하기 위해서 zero_grad를 해줬다.
    hypothesis = model(x_train)  # 현재까지는 순전파이다.
    loss = criterion(hypothesis, y_train)  # 1에포 상태
    
    loss.backward()   # 돌아가라 역전파 같음..
    optimizer.step()
    return loss.item()

EPOCHS = 100
for epoch in range(1, EPOCHS+1):
    loss = train(model, critenrion, optimizer, x_train, y_train)
    print("epoch: {}, loss: {:.8f}.".format(epoch, loss))   # "loss: {:.8f}  / 소수점 8번째까지만 출력해라"라는 뜻이다.



#4. 평가, 예측
print("=========== 평가, 예측 =============")
def evaluate(model, criterion, x_train, y_train): #그라디언티 적용 안해서 옵티마이저 필ㅇ없음? ??
    model.eval()
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = critenrion(hypothesis, y_test)
    return loss.item()

loss = evaluate(model, critenrion, x_test, y_test)
print("loss : ", loss)

y_predict = torch.argmax(model(x_test), axis=1)
print(y_predict[0:10])

from sklearn.metrics import r2_score
score = r2_score(y_test.cpu(), y_predict.cpu())
print("r2_score : ", score)

# loss :  31463.380859375
# tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
# r2_score :  -1.1001708513378987