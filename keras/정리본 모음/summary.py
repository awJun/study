"""=[ summary 사용 ]==============================================================================================

# Model: "sequential"
# _________________________________________________________________       
# Layer (type)                 Output Shape              Param #       
# =================================================================       
# dense (Dense)                (None, 5)                 10
# _________________________________________________________________       
# dense_1 (Dense)              (None, 3)                 18
# _________________________________________________________________       
# dense_2 (Dense)              (None, 4)                 16
# _________________________________________________________________       
# dense_3 (Dense)              (None, 2)                 10
# _________________________________________________________________       
# dense_4 (Dense)              (None, 1)                 3
# =================================================================       
# Total params: 57
# Trainable params: 57
# Non-trainable params: 0
# _________________________________________________________________ 

# 파이썬의 데이터 집계 함수 중의 하나인 model.summary()를 해서 레이어와 레이어 간의 연산 과정을 출력 해보면
# 뭔가 이상한 점을 느낄 수 있음
# input_dim이 1개고 바로 밑의 레이어의 노드 개수는 5개인데 Parameter 결과는 10이 나옴
# 1*5는 5인데 왜 10이 나왔을까?
# 그 밑에 과정들도 똑같음
# 5*3은 15인데 18이 나오고, 3*4는 12인데 16이 나옴 ... 
# 사실 각 노드와 노드간의 연산을 할 때 y = wx + b가 아님
# 각 노드와 노드간의 연산에서는 y = wx 밖에 없고, bias는 각 레이어에 노드개수 +1 의 형태로 숨어들어 있다고 볼 수 있음
# 각 레이어의 노드에 사실 bias 가 숨어 있어서 bais 까지 +1을 해줘야 진짜 노드의 개수가 완성된다고 볼 수 있음 (input_dim부터)
# 쉽게 생각해서 bias 값 때문에 원래 노드 개수에 1씩 더해서 다음레이어의 원래 노드 개수랑 곱한다고 보면 됨

========================================================================================================================
"""   



#2 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.summary()
