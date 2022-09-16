
# #1. 데이터
# # x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# # y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# # 완벽하게 정제된 데이터이므로 평가와 예측을 제대로 할 수 없다.

# x_train = np.array([1, 2, 3, 4, 5, 6, 7])
# x_test = np.array([8, 9, 10])

# y_train = np.array([1, 2, 3, 4, 5, 6, 7])
# y_test = np.array([8, 9, 10])


# ##### [11,12,13]을 예측하라!!!
# x_predict = np.array([11,12,13])

# 시작!!!
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()                   
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, '사용DVICE:', DEVICE)

#1. 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7])
x_test = np.array([8, 9, 10])

y_train = np.array([1, 2, 3, 4, 5, 6, 7])
y_test = np.array([8, 9, 10])


x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)


x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size())   # torch.Size([398, 30])
print(x_train.shape)    # torch.Size([398, 30])
# 둘 다 똑같은거임   size == shape  아무거나 사용해


#2. 모델
model = nn.Sequential(
    nn.Linear(7, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Linear()
).to(DEVICE)


#3. 컴파일, 훈련
critenrion = nn.BCELoss()    # # BCE : 바이너리크로스엔트로피   / 이진분류

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


y_predict = (model(x_test) >= 0.5).float()  # 숫자로 표현하기 위해서 float를 뒤에 붙힘
print(y_predict[0:10])   # 1. = 1  / 0. = 0

score = (y_predict == y_test).float().mean()   # 1이 7개고 0이 3개면 0.7 이것에 대한 평균을 내면 이게 acc이다.
print("accuracy : {:.4f}".format(score))










