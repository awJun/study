# logistic_regression.py  회기모델에 sigmoid를 붙힌 모델이다 즉, regression이지만 2진분류 모델이다.
# 넘파이로 변환할 떄 cpu로 변환해야한다. 아니면 cpg로 바꿔달라고 에러가 발생함
# 데이터가 gpu에서 만들어서 그렇다라고함

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


# #2. 모델
# model = nn.Sequential(
#     nn.Linear(30, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 1),
#     nn.Sigmoid(),
# ).to(DEVICE)


# 클래스 모델구성 완료
class Model(nn.Module):   # 괄호안에 있는 모듈을 상속하겠다. 즉, 상속시킬 것만 괄호안에 넣을 수 있음  /  나는 모듈안에 있는 변수와 함수를 사용하겟다
    def __init__(self, input_dim, output_dim):   # 초기화 / 정의 / 생성자 안에 있는 것을 실행을 시킨다. 무조건! 클래스에선
        # super().__init__()    # 다 상속해주세요 집주세요 차주세요 다 가져갈거에요   / nn.Module 안에 있는 것을 다쓰게 해준다.
        super(Model, self).__init__()    # 위랑 아래랑 똑같은거다 편한걸로 사용할 것.
        self.linear1 = nn.Linear(input_dim, 64)        # self. 나는 원래 모델출신이야 이 self의 linear이야 ~ 라고 선언해줘야 사용가능하다. 
        self.linear2 = nn.Linear(64, 32)                # self는 forward(순전파)에 들어갈 것을 미리 정의
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_size):    # forward : 순전파  파이토치는 forward 안에서 모델을 구성한다
        x = self.linear1(input_size)   # input_size 책에서는 그냥 x라고 표기된다. 하지만 통상적으로 처음에 input_size가 들어간다. 이름은 상황 없지만 1번 레이어의 괄호안과 통일만 시키면 된다.
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x   # x를 반환 ~
        
model = Model(30, 1).to(DEVICE)     # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat1 in method wrapper_addmm)
                                    # 우리는 GPU로 사용 할 것이므로 모델과 데이터에는 .to(DEVICE) 를 선언해줘야 한다.




#3. 컴파일, 훈련
criterion = nn.BCELoss()    # # BCE : 바이너리크로스엔트로피   / 이진분류

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



# accuracy : 0.9825
# accuracy_score :  0.9824561403508771