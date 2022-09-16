# 얘는 통으로 돌려서 안돌아ㅏㄹ 수 있음 안되면 안돼 라고 적을것
"""
[데이터가 커서 에러발생] 안돌아감
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.

(노드 아무리 줄여도 계속 터집니다..)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_covtype
from torch.utils.data import TensorDataset, DataLoader

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
print("torch : ", torch.__version__, "사용DEVICE : ", DEVICE)

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets["target"]

import numpy as np
print(np.unique(y))  # [1 2 3 4 5 6 7]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.7,
                                                    shuffle=True,
                                                    random_state=123,
                                                    # stratify=y <- 이거 키면 아래 에러 발생함
                                                    ) # ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.

x_train = torch.FloatTensor(x_train)
# y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)   # int가 길어지면 Long 
x_test = torch.FloatTensor(x_test)
# y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)  # Float가 커지면 doble
y_test = torch.LongTensor(y_test).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size())   # torch.Size([406708, 54])


train_set = TensorDataset(x_train, y_train)  # x와 y를 합쳐~
test_set = TensorDataset(x_test, y_test)  # x와 y를 합쳐~

print('================= len(train_set) ===================')
print(len(train_set))   # 406708

train_loader = DataLoader(train_set, batch_size=40000, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40000, shuffle=True)


# #2. 모델
# model = nn.Sequential(
#     nn.Linear(54, 64),   # 앞에는 컬럼 갯수
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 7),   # 마지막은 y의 유니크값 갯수 3개  # 클래스 갯 수에 맞춰서 해야한다.
#     nn.Sigmoid(),   # softmax를 안해줘도 된다.
# ).to(DEVICE)


class Model(nn.Module):    # Module을 사용할 때 forward를 사용안하면 에러발생 / NotImplementedError: Module [Model] is missing the required "forward" function
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 2)
        self.linear2 = nn.Linear(2, 2)
        self.linear3 = nn.Linear(2, 2)
        self.linear4 = nn.Linear(2, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()    # softmax를 안해줘도 된다.


    def forward(self, input_size):
        x = self.linear1(input_size)    
        x = self.relu(x)
        x = self.linear2(x) 
        x = self.relu(x)
        x = self.linear3(x) 
        x = self.relu(x)
        x = self.linear4(x) 
        x = self.sigmoid(x)
        return x

model = Model(54, 7).to(DEVICE)




#3. 컴파일, 훈련
# critenrion = nn.BCELoss()    # # BCE : 바이너리크로스엔트로피   / 이진분류
criterion = nn.CrossEntropyLoss()    # # BCE : 바이너리크로스엔트로피   / 이진분류   / CrossEntropyLoss : spars_crossEntropy  

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    # model. train()
    # for문은 이터레이터 형태로만 사용할 수 있다.  for문으로 돌리면 배치 크기로 빼준다.
    
    total_loss = 0   # 로스를 저장할 빈 리스트 생성
    
    for x_batch, y_batch in loader:  # 이렇게하면 알아서 x와 y로 잘려서 들어간다.
        optimizer.zero_grad()     # 역전파 이후 개인된 것으로 돌아갈 때 메모리에 잡것들이 남아있다고한다. 이것들을 제거하기 위해서 zero_grad를 해줬다.
        hypothesis = model(x_batch)  # 현재까지는 순전파이다.
        loss = criterion(hypothesis, y_batch)  # 1에포 상태
        
        loss.backward()   # 역전파
        optimizer.step()
        total_loss += loss.item()   # 1에포마다 계속 더해진다.
        
    return total_loss / len(loader)    # 전체 로스에서 loader의 갯수만큼 나눠준다.  

EPOCHS = 100
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, train_loader)
    print("epoch: {}, loss: {:.8f}.".format(epoch, loss))   # "loss: {:.8f}  / 소수점 8번째까지만 출력해라"라는 뜻이다.



#4. 평가, 예측
print("=========== 평가, 예측 =============")
def evaluate(model, criterion, loader): #그라디언티 적용 안해서 옵티마이저 필ㅇ없음? ??
    model.eval()
    total_loss = 0                # 로스를 담을 빈 변수 생성
    
    for x_batch, y_batch in loader:
        with torch.no_grad():
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            total_loss += loss.item()
            
    return loss.item()

loss = evaluate(model, criterion, x_test, y_test)
print("loss : ", loss)

y_predict = torch.argmax(model(x_test), axis=1)
print(y_predict[0:10])

score = (y_predict == y_test).float().mean()   # 1이 7개고 0이 3개면 0.7 이것에 대한 평균을 내면 이게 acc이다.
print("accuracy : {:.4f}".format(score))

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test.cpu(), y_predict.cpu())
print("accuracy_score : ", score)
