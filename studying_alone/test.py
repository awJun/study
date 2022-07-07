import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import load_digits

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from tensorflow.python.keras.models import Sequential, load_model  

#.1 데이터
datasets = load_digits()
x = datasets.data     # (1797, 64)
y = datasets.target   # (1797,)


from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
# print(y)
# print("+++++++++++++++++++++++++")
# print(x)

# print(x.shape) # (1797, 64)
# print(y.shape)   # (1797, 10)



# x: 17  /  y: 2
# print("y의 라벨값 : ", np.unique(x))


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )





scaler =  MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()



scaler.fit(x_train)
x_train = scaler.transform(x_train) # x_train을 수치로 변환해준다.
x_test = scaler.transform(x_test) # 
# print(np.min(x_train))   # 0.0
# print(np.max(x_train))   # 1.0000000000000002
# print(np.min(x_test))   # -0.06141956477526944
# print(np.max(x_test))   # 1.1478180091225068


loss, acc = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력
                                            # loss = loss / acc = metrics에서 나온 accuracy 값
print('loss : ', loss)
print('acc : ', acc)


result = model.evaluate(x_test, y_test)  # loss acc 각각 다른 리스트로 출력

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
print(y_predict)

y_test = np.argmax(y_test, axis=1)
# print(y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)

print("걸린시간 : ", end_time)

