# logistic_regression.py  회기모델에 sigmoid를 붙힌 모델이다 즉, regression이지만 2진분류 모델이다.

from calendar import EPOCH
from json import load
from os import scandir
from pickletools import optimize
from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
print("torch : ", torch.__version__, "사용DEVICE : ", DEVICE)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets["target"]

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.7,
                                                    shuffle=True,
                                                    random_state=123,
                                                    stratify=y
                                                    )
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
# 넘파이형태로 스케일링을 진행하기 때문에 x데이터는 스케일링 이후에 to(DEVICE)를 해야 에러가 안난다.


print(x_train.size())   # torch.Size([398, 30])
print(x_train.shape)    # torch.Size([398, 30])
# 둘 다 똑같은거임   size == shape  아무거나 사용해서 쓸 것


#2. 모델
model = nn.Sequential(
    nn.Linear(30, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid(),
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

# y_predict = model(x_test)
# print(y_predict[0:10])

# y_predict = (model(x_test) >= 0.5)
# print(y_predict[0:10])   # True = 1  / False = 0

y_predict = (model(x_test) >= 0.5).float()  # 숫자로 표현하기 위해서 float를 뒤에 붙힘
print(y_predict[0:10])   # 1. = 1  / 0. = 0

score = (y_predict == y_test).float().mean()   # 1이 7개고 0이 3개면 0.7 이것에 대한 평균을 내면 이게 acc이다.
print("accuracy : {:.4f}".format(score))  # accuracy : 0.9708


from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", score) # 에러    /  cpu로 변환해야함
# TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

score = accuracy_score(y_test.cpu(), y_predict.cpu())
print("accuracy_score : ", score)    # accuracy_score :  0.9766081871345029
# 위에와 동일한 결과를 얻었습니다. 살짝 다른 이유는 random_state가 랜덤으로 들어가서 그렇다.





