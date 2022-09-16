"""
[프로젝트 설명 링크]
https://colab.research.google.com/drive/18Pv_Ny0DP4rYPw-dxOoxojhGWBxuwvB6#scrollTo=qAz-H1C-aOVO


    # loss = criterion(hypothesis, y)
    # loss = nn.MSELoss().forward(hypothesis, y)
    # loss = nn.MSELoss()(hypothesis, y)
    loss = F.mse_loss(hypothesis, y)
3개중에서 아무거나 사용가능 이번에는 안에서 사용햅ㅎㅁ
"""


import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F   

USE_CUDA = torch.cuda.is_available()                   
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, '사용DVICE:', DEVICE)

# 1. data
x = np.array([1,2,3]) # (3, )
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE) # (3, 1)
y = torch.FloatTensor(y).unsqueeze(-1).to(DEVICE)

# 2. model
model = nn.Linear(1, 1).to(DEVICE) 

# 3. compile, fit
criterion = nn.MSELoss() # = loss
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    
    hypothesis = model(x)
    # loss = criterion(hypothesis, y)
    # loss = nn.MSELoss().forward(hypothesis, y)
    loss = nn.MSELoss()(hypothesis, y)
    # loss = F.mse_loss(hypothesis, y)
    
    loss.backward() 
    optimizer.step()

    return loss.item()
    
epochs = 20000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))
    
# 4. eval
def evaluate(model, criterion, x, y): 
    model.eval() 
    
    with torch.no_grad(): 
        x_predict = model(x)
        results = criterion(x_predict, y)
    
    return results.item()

result_loss = evaluate(model, criterion, x, y)
print(f'최종 loss: {result_loss}')

results = model(torch.Tensor([[4]]).to(DEVICE))   # 4를 예측  / .to(DEVICE)) 모델이랑 데이터는 .to(DEVICE))로 넘겨줘야한다. 여기서는 데이터이므로 넘겨줌
print(f'4의 예측값: {results.item()}')