"""=[ input_shape 사용 ]==============================================================================================

model  = Sequential()
# model.add(Dense(10, input_dim=3))      
# # model.summary()에서 (100, 3)  --> (None, 3)로 받아온다

- - - - - - - - - - - - - - - - 

model  = Sequential()
model.add(Dense(10, input_shape=(3,)))   
# 두 개는 같은 의미로 사용된다 / 차이점은 input_dim은 1차원 개념만 input_shape은 그 이상을 할 때
  사용된다.  input_shape를 사용 할 때 행은 무시하고 열만 괄호안에 넣어주면 된다.

--[ input_shape ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

model  = Sequential()
model.add(Dense(10, input_shape=(3,))) 
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

========================================================================================================================
"""  