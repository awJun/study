"""
[핵심]
레이어 동결을 레이어마다 부분적으로 하는 것이 포인트 



가중치를 초기화 안햇기 때문에 오히려 더 나쁨 ?


[해당 프로젝트 정리]
https://colab.research.google.com/drive/1hcqo_xwuheOojb7wovlze5ZEn3GRUKhe#scrollTo=tHlKy7IuAGQm
"""


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

# layer.trainable = False

# for layer in model.layers:
#     layer.trainable = False

model.layers[0].trainable = False   # Dense 0 번째 레이어 동결
# model.layers[1].trainable = False   # Dense 1 번째 레이어 동결
# model.layers[2].trainable = False   # Dense 2 번째 레이어 동결

model.summary()

# print(model.layers)  # 레이어에 관해서 출력해준다.
