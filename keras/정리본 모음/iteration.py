"""===[ iteration 내용]==================================================================================================================

# iteration의 의미
# 마지막으로 iteration은 1-epoch를 마치는데 필요한 미니배치 갯수를 의미합니다.
# 다른 말로, 1-epoch를 마치는데 필요한 파라미터 업데이트 횟수 이기도 합니다.
# 각 미니 배치 마다 파라미터 업데이터가 한번씩 진행되므로 iteration은 파라미터 업데이트 횟수이자 미니배치 갯수입니다.
# 예를 들어, 700개의 데이터를 100개씩 7개의 미니배치로 나누었을때, 1-epoch를 위해서는 7-iteration이 필요하며 7번의 파라미터 업데이트가 진행됩니다
# https://m.blog.naver.com/qbxlvnf11/221449297033

-[ iteration의 의미 추가설명 ]- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

전체 2000 개의 데이터가 있고, epochs = 20, batch_size = 500이라고 가정합시다.
그렇다면 1 epoch는 각 데이터의 size가 500인 batch가 들어간 네 번의 iteration으로 나누어집니다.
그리고 전체 데이터셋에 대해서는 20 번의 학습이 이루어졌으며, iteration 기준으로 보자면 총 80 번의 학습이 이루어진 것입니다.

===========================================================================================================================

# mse- mean(평균), squad(제곱) (오차) 음수가 상쇄되는 경우가 있기때문 , optimizer(최적화)
# batch = 데이터 하나씩 따로 작업하는것 *데이터가 많아질 때 overflow를 방지,batch size가 줄면 훈련 횟수 많아짐 단점은 시간이 오래걸림

========================================================================================================================
"""