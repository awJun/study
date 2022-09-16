from sklearn.datasets import load_diabetes
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
print("torch : ", torch.__version__, "사용DEVICE : ", DEVICE)

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets["target"]

# print(type(x)) <class 'numpy.ndarray'>

import numpy as np
# print(np.unique(y))  # 회귀

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.7,
                                                    shuffle=True,
                                                    random_state=123,
                                                    # stratify=y <- 이거 키면 아래 에러 발생함
                                                    ) # ValueError: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.

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

print(x_train.size())   # torch.Size([309, 10])


# 2. 모델
# model = nn.Sequential(
#     nn.Linear(10, 16),   # 앞에는 컬럼 갯수
#     nn.ReLU(),
#     nn.Linear(16, 16),
#     nn.ReLU(),
#     nn.Linear(16, 16),
#     nn.ReLU(),
#     nn.Linear(16, 1),   # 마지막은 y의 유니크값 갯수 3개  # 클래스 갯 수에 맞춰서 해야한다.
#     # nn.Linear(),   
# ).to(DEVICE)

class Model(nn.Module):   # 괄호안에 있는 모듈을 상속하겠다. 즉, 상속시킬 것만 괄호안에 넣을 수 있음  /  나는 모듈안에 있는 변수와 함수를 사용하겟다
    def __init__(self, input_dim, output_dim):   # 초기화 / 정의 / 생성자 안에 있는 것을 실행을 시킨다. 무조건! 클래스에선
        # super().__init__()    # 다 상속해주세요 집주세요 차주세요 다 가져갈거에요   / nn.Module 안에 있는 것을 다쓰게 해준다.
        super(Model, self).__init__()    # 위랑 아래랑 똑같은거다 편한걸로 사용할 것.
        self.linear1 = nn.Linear(input_dim, 32)        # self. 나는 원래 모델출신이야 이 self의 linear이야 ~ 라고 선언해줘야 사용가능하다. 
        self.linear2 = nn.Linear(32, 16)                # self는 forward(순전파)에 들어갈 것을 미리 정의
        self.linear3 = nn.Linear(16, 16)
        self.linear4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.Linear = nn.Linear(1, output_dim)
        
    def forward(self, input_size):    # forward : 순전파  파이토치는 forward 안에서 모델을 구성한다
        x = self.linear1(input_size)   # input_size 책에서는 그냥 x라고 표기된다. 하지만 통상적으로 처음에 input_size가 들어간다. 이름은 상황 없지만 1번 레이어의 괄호안과 통일만 시키면 된다.
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.Linear(x)
        return x   # x를 반환 ~
        
model = Model(10, 1).to(DEVICE)




#3. 컴파일, 훈련
# critenrion = nn.BCELoss()    # # BCE : 바이너리크로스엔트로피   / 이진분류
# critenrion = nn.CrossEntropyLoss()    # # BCE : 바이너리크로스엔트로피   / 이진분류   / CrossEntropyLoss : spars_crossEntropy  
criterion = nn.MSELoss()    # # BCE : 바이너리크로스엔트로피   / 이진분류   / CrossEntropyLoss : spars_crossEntropy  

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
    loss = train(model, criterion, optimizer, x_train, y_train)
    print("epoch: {}, loss: {:.8f}.".format(epoch, loss))   # "loss: {:.8f}  / 소수점 8번째까지만 출력해라"라는 뜻이다.



#4. 평가, 예측
print("=========== 평가, 예측 =============")
def evaluate(model, criterion, x_test, y_test): #그라디언티 적용 안해서 옵티마이저 필ㅇ없음? ??
    model.eval()
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
    return loss.item()

loss = evaluate(model, criterion, x_test, y_test)
print("loss : ", loss)

y_predict = torch.argmax(model(x_test), axis=1)
print(y_predict[0:10])

from sklearn.metrics import r2_score
score = r2_score(y_test.cpu(), y_predict.cpu())
print("r2_score : ", score)

# loss :  6214.95947265625
# tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
# r2_score :  -3.7178931254725933