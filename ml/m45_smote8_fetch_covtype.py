# 실습
# m45_7에서 저장한 데이터를 불러와서 
# 완성 및 성능 비교


import pickle
path = 'd:/study_data/_save/_xg/m45_smote7/'
x_train = pickle.load(open(path + 'x_train_save.dat', 'rb'))    # dump로 저장함
y_train = pickle.load(open(path + 'y_train_save.dat', 'rb'))    # dump로 저장함
x_test = pickle.load(open(path + 'x_test_save.dat', 'rb'))    # dump로 저장함
y_test = pickle.load(open(path + 'y_test_save.dat', 'rb'))    # dump로 저장함

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

#2. 모델  /  #3. 훈련
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

score = model.score(x_test, y_test)
print("model.score : ", score)

from sklearn.metrics import accuracy_score, f1_score
print('acc_score : ', accuracy_score(y_test, y_predict))
print("f1_score(macro) : ", f1_score(y_test, y_predict, average='macro'))
# print("f1_score(micro) : ", f1_score(y_test, y_predict, average='micro'))

# model.score :  0.960319440978288
# acc_score :  0.960319440978288
# f1_score(macro) :  0.9400253213139662