"""=[  ModelCheckpoint 설명 ]==============================================================================================

import datetime
date = datetime.datetime.now()
print(date) # 2022-07-07 17:24:51.433145  # 수치형 데이터이다.

date = date.strftime('%m%d_%H%M')  # %m : 월 / %d : 일 / %H : 시 / %M : 분
print(date) # 0707_1724            # 해석: 7월 7일 _ 17시 24분

filepath = './_ModelCheckPoint/k24'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#         {에포의 4자리}-{발로스의 소수점 4째자리} 라는 뜻

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
                      save_best_only=True,
                      filepath= "".join([filepath, 'k24_', date, '_', filename] # .join안에 있는 모든 문자열을 합치겠다.
                      ))




















===[ ModelCheckpoint 사용 ]==============================================================================================

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, # 가장 좋은 가중치 저장 위해 / mode가 모니터한 가장 최적 값, val 최저값, accuracy 최고값
                      save_best_only=True,      # 모니터 후 가장 좋은 값을 저장
                      filepath='./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5' # 가장 낮은 지점이 이 경로에 저장, 낮은 값이 나올 때마다 계속적으로 갱신하여 저장
                      )

# start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=20, 
                validation_split=0.2,
                callbacks=[earlyStopping, mcp], # 최저값을 체크해 반환해줌
                verbose=1)
# end_time = time.time()
'''

model = load_model('./_ModelCheckPoint/keras24_ModelCheckPoint.hdf5')  # mcp에서 val_loss가 가장 최저값 상태일때의
                                                                         가중치 값을 불러옴

========================================================================================================================
"""   