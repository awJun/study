"""=[ 함수형 모델 사용 ]==============================================================================================

### 새로운 모델 ###
input1 = Input(shape=(3,))   # 처음에 Input 명시하고 Input 대한 shape 명시해준다.
dense1 = Dense(10)(input1)   # Dense 구성을하고  node 값을 넣고 받아오고 싶은 변수 받아온다.
dense2 = Dense(5, activation = 'relu')(dense1)    # 받아온 변수를 통해 훈련의 순서를 사용자가 원하는대로 할 수 있다.
dense3 = Dense(3, activation = 'sigmoid')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1) # 해당 모델의 input과 output을 설정한다.

Input(shape=(3,)  <--- input_dim=3과 같음
========================================================================================================================
"""  
