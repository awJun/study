from pickletools import optimize
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,7])


#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))


#3. 컴파일, 훈련
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop, nadam

learning_rate = 0.1

optimizer_Adam = adam.Adam(learning_rate=learning_rate)
optimizer_Adadelta = adadelta.Adadelta(learning_rate=learning_rate)
optimizer_Adagrad = adagrad.Adagrad(learning_rate=learning_rate)
optimizer_Adamax = adamax.Adamax(learning_rate=learning_rate)
optimizer_RMSprop = rmsprop.RMSprop(learning_rate=learning_rate)
optimizer_Nadam = nadam.Nadam(learning_rate=learning_rate)
# model.compile(loss="mse", optimizer="adam")

optimizer_list = [optimizer_Adam, optimizer_Adadelta, optimizer_Adagrad, optimizer_Adamax, optimizer_RMSprop, optimizer_Nadam]


for i in optimizer_list:
    
    model.compile(loss="mse", optimizer=i)   # 텐서2에서 adam과 함께 learning_rate까지 사용가능하게 작업함   

    model.fit(x, y, epochs=50, batch_size=1)

    #4. 평가, 예측
    loss = model.evaluate(x, y)
    y_predict = model.predict([11])

    optimizer_name = i.__class__.__name__
    
    print(optimizer_name, "loss : ", round(loss, 4), "lr : ", learning_rate, "결과물 : ", y_predict)


# models = [LinearSVC, SVC, Perceptron, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]

# for model in models:
#     model = model()
#     model_name = str(model).strip('()')  # .strip('()')참고 https://ai-youngjun.tistory.com/68
#     model.fit(x_train, y_train)
#     result = model.score(x_test, y_test)
#     print(model_name, '결과: ', result)











