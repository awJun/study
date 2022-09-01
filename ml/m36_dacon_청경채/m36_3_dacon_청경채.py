import pandas as pd
import numpy as np
import glob

path = './_data/dacon_Bok/'
all_input_list = sorted(glob.glob(path + 'train_input/*.csv'))
all_target_list = sorted(glob.glob(path + 'train_target/*.csv'))

train_input_list = all_input_list[:50]
train_target_list = all_target_list[:50]

val_input_list = all_input_list[50:]
val_target_list = all_target_list[50:]

# print(all_input_list)
print(val_input_list)
print(len(val_input_list))  # 8

def aaa(input_paths, target_paths): #, infer_mode):
    input_paths = input_paths
    target_paths = target_paths
    # self.infer_mode = infer_mode
   
    data_list = []
    label_list = []
    print('시작...')
    # for input_path, target_path in tqdm(zip(input_paths, target_paths)):
    for input_path, target_path in zip(input_paths, target_paths):
        input_df = pd.read_csv(input_path)
        target_df = pd.read_csv(target_path)
       
        input_df = input_df.drop(columns=['시간'])
        input_df = input_df.fillna(0)
       
        input_length = int(len(input_df)/1440)
        target_length = int(len(target_df))
        print(input_length, target_length)
       
        for idx in range(target_length):
            time_series = input_df[1440*idx:1440*(idx+1)].values
            # self.data_list.append(torch.Tensor(time_series))
            data_list.append(time_series)
        for label in target_df["rate"]:
            label_list.append(label)
    return np.array(data_list), np.array(label_list)
    print('끗.')

x_train, y_train = aaa(train_input_list, train_target_list) #, False)
x_test, y_test = aaa(val_input_list, val_target_list) #, False)

# print(x_train.shape)  # (1607, 1440, 37)
# print(y_train.shape)  # (1607,)
# print(x_test.shape)   # (206, 1440, 37)
# print(y_test.shape)   # (206,)

x_train = x_train.reshape(1607, 1440 * 37)   
x_test = x_test.reshape(206, 1440 * 37)
print(x_train.shape)  # (1607, 53280)
print(x_test.shape)   # (206, 53280)

# print(x_train[0])
# print(len(x_train), len(y_train)) # 1607 1607
# print(len(x_train[0]))   # 1440
# print(y_train)   # 1440
# print(x_train.shape, y_train.shape)   # (1607, 1440, 37) (1607,)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=1000)

# x_train = pca.fit_transform(x_train) 
# x_test = pca.fit_transform(x_test) 
# print(x_train.shape)   # (1607, 12)
# print(x_test.shape)    # (2019, 12)

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=1000,
              learning_rate=1,
              max_depth=2,
              gamma=0,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=0.5,
              colsample_bylevel=1,
              colsample_bynode=1,
              reg_alpha=0.01,
              reg_lambd=1,
              tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234,
              )

# 3. 훈련
import time

start = time.time()
model.fit(x_train, y_train, early_stopping_rounds=500,     # early_stopping_rounds=10 : 10번까진 참겠다.
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric=['logloss'],
          )
end = time.time() -start

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('결과: ', results)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 

# print(model, ': ', model.feature_importances_) # tree계열에만 있음


import os
import zipfile
filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
os.chdir("D:/study_data/_data\dacon_Bok/sample_submission")
with zipfile.ZipFile("submission.zip", 'w') as my_zip:
    for i in filelist:
        my_zip.write(i)
    my_zip.close()



###[ r2_score ]###########################
# 결과:  0.8070573257161597











