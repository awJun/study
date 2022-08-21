#.1 데이터 불러오기
# import pandas as pd

# path = "D:/study_data/_data/dacon_Bok/"

# x_train = pd.read_csv(path + "train_input/input_train.csv", index_col=None, header=0, sep=',')
# y_train = pd.read_csv(path + "train_target/target_train.csv", index_col=None, header=0, sep=',')
# x_test = pd.read_csv(path + "test_input/input_test.csv", index_col=None, header=0, sep=',')
# y_test = pd.read_csv(path + "test_target/target_test.csv", index_col=None, header=0, sep=',')


# TEST_01

# for i in range(8):


# .1 데이터 불러오기
import pandas as pd

###[ train_input 불러오기 ]####################################
x_train_path = "D:/study_data/_data/dacon_Bok/train_input/"
def x_trains(aaa):
    for i in range(1, 58):
        pd.read_csv(x_train_path + "CASE_" + aaa + ".csv", index_col=None, header=0, sep=',')




























