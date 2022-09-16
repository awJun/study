# x와 y를 합쳐서 한 번에 던지는 것 이것이 DataLoader이라고한다.
"""
[해당 프로젝트 설명]
https://colab.research.google.com/drive/1nWNFARGbj7m-CrFW5dIXwLNijuMmFaab#scrollTo=lkyuUqjouDrL

"""



from enum import unique
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

################################# 요기서 시작 #####################################
from torch.utils.data import TensorDataset, DataLoader  # x, y를 합친것 / 거기에 배치까지 한칩것
train_set = TensorDataset(x_train, y_train)  # x와 y를 합쳐~
test_set = TensorDataset(x_test, y_test)  # x와 y를 합쳐~

print(train_set)  # <torch.utils.data.dataset.TensorDataset object at 0x00000243628268B0>
print("========================================== [ train_set[0] ] ===========================================")
print(train_set[0])   #  첫 행의 데이터 + 해당 행의 유니크 값
print("========================================== [ train_set[0][0] ] ===========================================")
print(train_set[0][0])   # 첫 행의 데이터
print("========================================== [ train_set[0][1] ] ===========================================")
print(train_set[0][1])   # 해당 행의 유니크 값
# print(train_set[0][1])   # [0] x_train에 대한 데이터가 추출된다 0번째 행!
print(len(train_set))   # 398

train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=True)


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

def train(model, criterion, optimizer, loader):
    # model. train()
    # for문은 이터레이터 형태로만 사용할 수 있다.  for문으로 돌리면 배치 크기로 빼준다.
    
    total_loss = 0   # 로스를 저장할 빈 리스트 생성
    
    for x_batch, y_batch in loader:  # 이렇게하면 알아서 x와 y로 잘려서 들어간다.
        optimizer.zero_grad()     # 역전파 이후 개인된 것으로 돌아갈 때 메모리에 잡것들이 남아있다고한다. 이것들을 제거하기 위해서 zero_grad를 해줬다.
        hypothesis = model(x_batch)  # 현재까지는 순전파이다.
        loss = criterion(hypothesis, y_batch)  # 1에포 상태
        
        loss.backward()   # 돌아가라 역전파 같음..
        optimizer.step()
        total_loss += loss.item()   # 1에포마다 계속 더해진다.
        
    return total_loss / len(loader)    # 전체 로스에서 loader의 갯수만큼 나눠준다.   len으로 나누기는 필수사항은 아님 어차피 자기자신이라 비교하는 용도이기 때문에  
                                                                                    # 그냥 여러개를 보고 얼마나 갱신되엇느닞 볼 수 잇다 하지만 좀 더 정확하게 보기위해서 이작을했다.

# 1 배치 작업에선 훈련이 10번?   30개의 배치 ?

EPOCHS = 100
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, train_loader)
    if epoch % 10 ==0:
        print("epoch: {}, loss: {:.8f}.".format(epoch, loss))   # "loss: {:.8f}  / 소수점 8번째까지만 출력해라"라는 뜻이다.

#4. 평가, 예측
print("=========== 평가, 예측 =============")
def evaluate(model, criterion, loader): #그라디언티 적용 안해서 옵티마이저 필ㅇ없음? ??
    model.eval()
    total_loss = 0
    
    for x_batch, y_batch in loader:
        with torch.no_grad():
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            total_loss += loss.item()
            
    return loss.item()

loss = evaluate(model, criterion, test_loader)
print("loss : ", loss)

# y_predict = model(x_test)
# print(y_predict[0:10])

# y_predict = (model(x_test) >= 0.5)
# print(y_predict[0:10])   # True = 1  / False = 0

# predict에 있는 input shape는 변동사항 없음

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