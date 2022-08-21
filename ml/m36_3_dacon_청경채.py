# # 데이터 작업
# import os
# import pandas as pd
# import glob

# def merge_excel_files(file_path, file_format, save_path, save_format, columns=None):
#     merge_df = pd.DataFrame()
#     file_list = file_list = [f"{file_path}/{file}" for file in os.listdir(file_path) if file_format in file]
    
#     for file in file_list:
#         if file_format == ".csv":
#             file_df = pd.read_csv(file)
     
#         if columns is None:
#             columns = file_df.columns
            
#         temp_df = pd.DataFrame(file_df, columns=columns)
        
#         merge_df = merge_df.append(temp_df)
        
#     if save_format == ".csv":
#         merge_df.to_csv(save_path, index=False)
        

# if __name__ == "__main__":
#     merge_excel_files(file_path="D:/study_data/_data/dacon_Bok/train_target", 
#                       save_path="D:/study_data/_data/dacon_Bok/train_target/청경채.csv",
#                       file_format=".csv",
#                       save_format=".csv"
#                       )

#.1 데이터 불러오기
import pandas as pd

path = "D:/study_data/_data/dacon_Bok/"

x_train = pd.read_csv(path + "train_input/input_train.csv", index_col=None, header=0, sep=',')
y_train = pd.read_csv(path + "train_target/target_train.csv", index_col=None, header=0, sep=',')
x_test = pd.read_csv(path + "test_input/input_test.csv", index_col=None, header=0, sep=',')
y_test = pd.read_csv(path + "test_target/target_test.csv", index_col=None, header=0, sep=',')

# print(x_train.head(5))   # head 디폴트는 5
# print(y_train.head(5))   # head 디폴트는 5
# print(x_test.head(5))   # head 디폴트는 5
# print(y_test.head(5))   # head 디폴트는 5



#.2 날짜 데이터 분리
###[ trian 분리 ]################################################
# print(x_train.shape)  # (2611507, 38)

x_train['일자'] = pd.to_datetime(x_train['시간'])
x_train['연도'] = x_train['일자'].dt.year
x_train['월'] = x_train['일자'].dt.month
x_train['일'] = x_train['일자'].dt.day

# print(x_train.shape)  # (2611507, 42)
###[ test 분리 ]##################################################
# print(x_test.shape)  # (285120, 38)

x_test['일자'] = pd.to_datetime(x_test['시간'])
x_test['연도'] = x_test['일자'].dt.year
x_test['월'] = x_test['일자'].dt.month
x_test['일'] = x_test['일자'].dt.day

# print(x_test.shape)   # (285120, 42)


#.3 train, test 결측치 여부확인  (너무 많은 것을 확인..)
###[ 결측치 여부확인 ]############################################
# print(x_train.isnull().sum())   
# print(x_test.isnull().sum())   

###[ 결측치 처리 ]#################################################
# x_train = x_train.dropna(axis=0)
# x_test = x_test.dropna(axis=0)
# print(x_train.isnull().sum())   
# print(x_train.shape)  # (1376042, 42)

# print(x_test.isnull().sum())   
# print(x_test.shape)  # (129591, 42)



# print(x_train.loc[41760:])
























