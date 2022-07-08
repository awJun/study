#  ModelCheckpoint

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

filepath = './_ModelCheckPoint/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   
# {epoch:04d} epoch를 4자리까지 제한을 하겠다. 
# {val_loss:.4f} val_loss 뒤에 소수점을 4자리까지로 제한하겠다.      

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([filepath, 'k24_', date, '_', filename])
                      ) # filepath= ""  -->  ""를 사용해서 빈공간 생성
                        # join()안에 항목을 합쳐서 출력하겠다.       
                        # k24_도 출력될 때 나오는 거임 그냥 이름임 이건 굳이 안넣어도 됨
# save_best_only=True, 디폴트 값은 false True로 지정해서 사용할 것.
# save_best_only=True는 훈련과정을 출력해주다가 값이 뇌절 치는 구간은 자르고 다시 잘나오는 
# 구간부터 다시 출력해준다

hist = model.fit(x_train, y_train, epochs=100, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)




# save 관련 정보
# https://keras.io/api/models/model_saving_apis/

###########################################################
model.save("./_save/keras23_3_save_model.h5")
# 모델에 관련 정보를 해당 경로에 세이브 해둠
# #2. 마지막 단에 쓰면 #2.까지 세이브되고 / #3. 마지막 단에 쓰면 #3. 까지 세이브 해준다.

model = load_model("./_save/keras23_3_save_model.h5")
# 해당 경로에 있는 모델을 가져옴
# 해당 경로에 있는 저장된 모델과 가중치 값을 가져온다
# 해당 모델을 만들 때 나왔던 가중치 값이 그대로 저장되고 그 값을 가져옴
###########################################################
 #       [ HDF5 형식을 사용한다. ]
model.save_weights("./_save/keras23_5_save_weights1.h5")
# 해당 모델의 레이어 가중치를 저장합니다.

model.load_weights("./_save/keras23_5_save_weights2.h5")
# 해당 모델의 레이어 가중치를 로드합니다.

model.get_weights("./_save/keras23_5_save_weights2.h5")
# 해당 모델의 가중치를 검색합니다.

model.set_weights("./_save/keras23_5_save_weights2.h5")
# NumPy 배열에서 레이어의 가중치를 설정합니다.
###########################################################

model.get_config("./_save/keras23_5_save_weights2.h5")
# 모델의 구성을 반환합니다

model.from_config("./_save/keras23_5_save_weights2.h5")
# 모델의 구성에서 레이어를 생성합니다.
###########################################################




