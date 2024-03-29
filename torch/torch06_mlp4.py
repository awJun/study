import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()                   
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, '사용DVICE:', DEVICE)

# 1. data
x = np.array([range(10), range(21, 31), range(201,211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1.,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,8,7,6,5,4,3,2,1,0]])
x_test = np.array([9, 30, 210])
x_test = x_test.reshape(-1, 3)

x = torch.FloatTensor(np.transpose(x)).to(DEVICE)
y = torch.FloatTensor(np.transpose(y)).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

###### 스케일링 ######
x_test = (x_test - torch.mean(x)) / torch.std(x) 
x = (x - torch.mean(x)) / torch.std(x)

print(x.shape, y.shape, x_test.shape)

# 2. model
model = nn.Sequential(
    nn.Linear(3,4),
    nn.Linear(4,5),
    nn.Linear(5,3),
    nn.ReLU(),          # 위에 적용됨
    nn.Linear(3,2),
    nn.Linear(2,3)
).to(DEVICE)

# 3. compile, fit
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, optimizer, x, y):
    optimizer.zero_grad()
    
    hypothesis = model(x)
    loss = F.mse_loss(hypothesis, y)
    
    loss.backward() 
    optimizer.step()

    return loss.item()
    
epochs = 3
for epoch in range(1, epochs+1):
    loss = train(model, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))
    
# 4. eval
def evaluate(model, x, y): 
    model.eval() 
    
    with torch.no_grad(): 
        x_predict = model(x)
        results = nn.MSELoss()(x_predict, y)
    
    return results.item()

result_loss = evaluate(model, x, y)
print(f'최종 loss: {result_loss}')

results = model(x_test)
# results = results.cpu().detach().numpy()
print(f'예측값: {results.tolist()}')

# 나와야할 값 : 10, 1.9, 0

# 최종 loss: 4.319656177892428e-12
# 예측값: [ 1.0000e+01,  1.9000e+00, -1.8179e-06]