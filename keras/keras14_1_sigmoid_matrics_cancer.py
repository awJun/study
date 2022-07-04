import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import time
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer

from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR )
# # (569, 30)
# print(datasets.feature_names)

x = datasets.data  # [data] = /data 안에 키 벨류가 있으므로 똑같은 형태다.
y = datasets.target 




# print(x.shape, y.shape) (569, 30) (569,)  y는 569개

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100
                                                    )




#2. 모델구성
model = Sequential()    
model.add(Dense(100, activation='relu', input_dim=30))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일. 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam')

model.compile(loss='binary_crossentropy', optimizer='adam',
             metrics=['accuracy']) # loss에 accuracy도 같이 보여달라고 선언한 것이다. 
                                   # metrics=[]를 사용하면 로스 외에 다른 지표도 같이 출력해준다.
            # 컴파일 안에 metrics는 평가지표라고도불린다.
    # 여기서 mse는 회귀모델에서 사용하는 활성화 함수이므로 분류모델에서는 신용성은 없다. 
    
    
from tensorflow.python.keras.callbacks import EarlyStopping   # early: "빨리"라는 뜻
earlyStopping = EarlyStopping(monitor='val_loss', patience=1000, mode='auto', verbose=1, 
                              restore_best_weights=True) 

                # restore_best_weights Treu로 설정하면 monitor하고 있던 값중에 가장 좋았던
                # weight 값으로 복원후에 가져와서 weight로 사용한다.
                # restore_best_weights False로 설정하면 중단된 시점의 weight 값을 사용한다.
                
                
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=50,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time()


#4. 평가, 예측
y_predict = model.predict(x_test)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = y_predict.round(0)    
                                  # 여기에서의 round는 .round이므로 반올림을 해주며 0  or 1로 
                                  # 출력하면서 자리수를 제한해주는 역할로 사용되는것 같음
                                  # 여기에서 .round를 사용함으로써 위에서 sigmoid에서 0 ~ 1 사이
                                  # 의 값으로 출력된 실수 데이터를 2진으로 변환하여 분류모델에서
                                  # 지장없이 사용할 수 있는 역할을 해준다.
                                  
                                  # round(0.2) # 0 반환
                                  # round(0.7) # 1 반환
                                  # round(1.6) # 2 반환
                                  # 파이썬의 round 함수는 똑같이 가깝다면 짝수를 반환
                                  # ex) 0.5  -> 0    /    2.5  ->  2      /     1.5  ->   2  




## 과제: 아래 accuracy 스코어 완성
from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)

print("걸린시간 : ", end_time)



import matplotlib.pyplot as plt
plt.figure(figsize = (9,6))     # figsize(가로길이,세로길이)
plt.plot(hist.history['loss'], marker='.', color='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', color='blue', label='val_loss')
plt.grid(True, axis=('x'))                      # plt.grid(True)
plt.title('asaql')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')   # 그래프가 없는쪽에 알아서 해준다 굳이 명시할 필요 없음
# plt.legend()                    #   <-- 명시 안한상태
plt.show()


# 현재 모델은 sigmoid를 사용해서 분류모델로 만들었기 때문에 결정계수를 구하는 r2_score는 사용불가능

"""     [초기형태]
[데이터]
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100
                                                    )
                                                    
------------------------------------------------------------------------
[모델구성]
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=30))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

------------------------------------------------------------------------
[EarlyStopping]
earlyStopping = EarlyStopping(monitor='val_loss', patience=1000, mode='auto', verbose=1, 
                              restore_best_weights=True) 

------------------------------------------------------------------------
[컴파일]
model.compile(loss='binary_crossentropy', optimizer='adam')

------------------------------------------------------------------------

[훈련]
fit(x_train, y_train, epochs=10000, batch_size=50,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
                 
"""
# 결과값 1회차
# loss :  0.1261337846517563
# accuracy :  0.935672514619883
# 걸린시간 :  1656744100.296302

# 결과값 2회차
# loss :  0.12038397043943405
# accuracy :  0.935672514619883
# 걸린시간 :  1656745168.190372

# 결과값 3회차
# loss :  0.12472661584615707
# accuracy :  0.9415204678362573
# 걸린시간 :  1656745513.3523018
"""    
[모델구성]      [activation='sigmoid'만 사용]
model = Sequential()
model.add(Dense(100, activation='sigmoid', input_dim=30))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
              
"""
# 결과값: 1회차
# loss :  0.1118834987282753
# accuracy :  0.9649122807017544
# 걸린시간 :  1656744468.481469

# 결과값: 2회차
# loss :  0.14029723405838013
# accuracy :  0.9649122807017544
# 걸린시간 :  1656744663.0738435

# 결과값: 3회차
# loss :  0.12987937033176422
# accuracy :  0.9473684210526315
# 걸린시간 :  1656744789.1705852

"""
[모델구성]      [activation='relu'만 사용]
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=30))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

"""
# 결과값 1회차
# loss :  0.0673321932554245
# accuracy :  0.9824561403508771
# 걸린시간 :  1656745728.033523

# 결과값 2회차
# loss :  0.0862422063946724
# accuracy :  0.9707602339181286
# 걸린시간 :  1656745867.6411462

# 결과값 3회차
# loss :  0.08734232932329178
# accuracy :  0.9707602339181286
# 걸린시간 :  1656745975.7435894"



"""
[최종평가]
activation을 linear, relu, sigmoid를 사용했을 때와 activation을 'sigmoid'만 사용했을 때는 성능에
큰 차이는 없었지만 activation을 'relu'만 사용했을 때는 기존과는 다르게 성능이 훨씬 좋아졌다.

"""





