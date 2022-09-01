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



# reg = ak.TextRegressor(overwrite=True, max_trials=1)





















# import os
# import zipfile
# filelist = ['TEST_01.csv','TEST_02.csv','TEST_03.csv','TEST_04.csv','TEST_05.csv', 'TEST_06.csv']
# os.chdir("D:/study_data/_data\dacon_Bok/sample_submission")
# with zipfile.ZipFile("submission.zip", 'w') as my_zip:
#     for i in filelist:
#         my_zip.write(i)
#     my_zip.close()



