from numpy.lib.npyio import load
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
import pickle

file_path = '/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/saved_pickle/cleanTuGong_car2.pkl'
with open(file_path, 'rb') as f:
    sliced_result = pickle.load(f)

color_str = ['k', 'b', 'r', 'g', 'y']
color_str = ['k', 'k', 'k', 'k', 'k']
plot_num = 5

plt.figure(1)
plt.clf()
plt.cla()

saved_list = []
for i in range(plot_num):
    saved_list.append(sliced_result[i+1])
    tmp_df = sliced_result[i+1]
    if(i == 0):
        plt.plot(tmp_df.iloc[0]['x'], tmp_df.iloc[0]['y'],
                 color=color_str[i % 5], marker='o')
    if(i == plot_num-1):
        plt.plot(tmp_df.iloc[-1]['x'], tmp_df.iloc[-1]
                 ['y'], color=color_str[i % 5], marker='o')
    # plt.scatter(tmp_df['x'], tmp_df['y'], c = color_str[i % 5])
    plt.plot(tmp_df['x'], tmp_df['y'], color_str[i % 5])

with open('/home/Gong.gz/Public_Dataset/ncbdc2021/testSliced.pkl','wb') as f:
    pickle.dump(saved_list,f)
# save_file = '/home/Gong.gz/HDD/2021-NCBDC/figure/切分行驶片段可视化/fig2.jpg'
# plt.title("Slice Path")
# plt.xlabel("X/m")
# plt.ylabel("Y/m")
# plt.savefig(save_file, dpi=600)
