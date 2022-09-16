"""
텐서로 변경해주는 역할
x = torch.FloatTensor(x)  
y = torch.FloatTensor(y)

"""

# logistic_regression.py  회기모델에 sigmoid를 붙힌 모델이다 즉, regression이지만 2진분류 모델이다.
# squeeze 1을 없애겟다  / unsqueeze는 확장

from sklearn.datasets import load_iris
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")
print("torch : ", torch.__version__, "사용DEVICE : ", DEVICE)

#1. 데이터
datasets = load_iris()
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
# 넘파이형태로 스케일링을 진행하기 때문에 x데이터는 스케일링 이후에 to(DEVICE)를 해야 에러가 안난다.


print(x_train.size())   # torch.Size([150, 4])
print(x_train.shape)    # torch.Size([150, 4])
# 둘 다 똑같은거임   size == shape  아무거나 사용해서 쓸 것


# #2. 모델
# model = nn.Sequential(
#     nn.Linear(4, 64),   # 앞에는 컬럼 갯수
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 3),   # 마지막은 y의 유니크값 갯수 3개  # 클래스 갯 수에 맞춰서 해야한다.
#     # nn.Sigmoid(),   # softmax를 안해줘도 된다.
# ).to(DEVICE)

class Model(nn.Module):    # Module을 사용할 때 forward를 사용안하면 에러발생 / NotImplementedError: Module [Model] is missing the required "forward" function
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 16)
        self.linear4 = nn.Linear(16, output_dim)
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

model = Model(4, 3).to(DEVICE)

#3. 컴파일, 훈련
# critenrion = nn.BCELoss()    # # BCE : 바이너리크로스엔트로피   / 이진분류
criterion = nn.CrossEntropyLoss()    # # BCE : 바이너리크로스엔트로피   / 이진분류   / CrossEntropyLoss : spars_crossEntropy  

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
def evaluate(model, criterion, x_train, y_train): #그라디언티 적용 안해서 옵티마이저 필ㅇ없음? ??
    model.eval()
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
    return loss.item()

loss = evaluate(model, criterion, x_test, y_test)
print("loss : ", loss)



y_predict = torch.argmax(model(x_test), axis=1)
print(y_predict[0:10])

# y_predict = (model(x_test) >= 0.5)
# print(y_predict[0:10])   # True = 1  / False = 0

# y_predict = (model(x_test) >= 0.5).float()  # 숫자로 표현하기 위해서 float를 뒤에 붙힘
# print(y_predict[0:10])   # 1. = 1  / 0. = 0

score = (y_predict == y_test).float().mean()   # 1이 7개고 0이 3개면 0.7 이것에 대한 평균을 내면 이게 acc이다.
print("accuracy : {:.4f}".format(score))  # accuracy : 0.9708


from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", score) # 에러    /  cpu로 변환해야함
# TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

score = accuracy_score(y_test.cpu(), y_predict.cpu())
print("accuracy_score : ", score)    # accuracy_score :  0.9766081871345029
# 위에와 동일한 결과를 얻었습니다. 살짝 다른 이유는 random_state가 랜덤으로 들어가서 그렇다.




# loss :  0.6129800081253052
# tensor([2, 2, 1, 1, 2, 0, 0, 2, 0, 0], device='cuda:0')
# accuracy : 0.9333
# accuracy_score :  0.9333333333333333
