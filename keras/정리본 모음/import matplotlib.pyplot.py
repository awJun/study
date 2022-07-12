import matplotlib.pyplot as plt

plt.figure(figsize = (9,6))     # figsize(가로길이,세로길이)
plt.plot(hist.history['loss'], marker='.', color='red', label='loss')           # label='loss' 해당 선 이름
plt.plot(hist.history['val_loss'], marker='.', color='blue', label='val_loss')  # marker='.' 점으로 찍겟다
plt.grid()                        # plt.grid(True)    # grid: 그리다
plt.title('asaql')                # title의 이름을 asaql로 하겠다
plt.ylabel('loss')                # y라벨의 이름을 loss로 하겠다
plt.xlabel('epochs')              # x라벨의 이름을 epochs로 하겠다
plt.legend(loc = 'upper right')   # 그래프가 없는쪽에 알아서 해준다 굳이 명시할 필요 없음
# plt.legend()   # 그래프가 없는쪽에 알아서 해준다 굳이 명시를 안 할 경우 사용법
plt.show()    # 그래프를 보여줘라