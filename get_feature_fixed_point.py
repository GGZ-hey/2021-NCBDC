from numpy.lib.npyio import load
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
import pickle
from one_hot_by_hour import *


def dropNAn(sliced_df, drop_name_list):
    sliced_df = sliced_df.dropna(subset=drop_name_list)
    return sliced_df


car_num = 1
fixed_point_num = 30
feature_list = []  # 一个片段30个点，还要drop掉空值
drop_name_list = ['车速', '驱动电机转矩', '累计里程', '驱动电机转速',
                  '驱动电机温度', '总电压', '总电流', '经度', '维度', '最高温度值', '最低温度值', '时间']

# for i in range(car_num):
i = 9
filename = '/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/saved_pickle/TuXkClean_car' + \
    str(i) + '.pkl'
with open(filename, 'rb') as read_f:  # 用with的优点是可以不用写关闭文件操作
    sliced_result_ffile = pickle.load(read_f)
# sliced_result_ffile -> List
for j in range(len(sliced_result_ffile)):  # 每个行驶片段
    # if(sliced_result_ffile[j]['总电流'].isnull().any()): ### TEST
    #     print(j)

    tmp_sliced_df = sliced_result_ffile[j].copy(deep=True)
    if(type(tmp_sliced_df) == pd.Series):
        tmp_sliced_df = pd.DataFrame(tmp_sliced_df.values.reshape(
            1, -1), columns=tmp_sliced_df.index.to_list())
    # print("第{}辆车，第{}个片段".format(i, j))
    # print('Before Drop: ', tmp_sliced_df.shape)
    tmp_sliced_df = dropNAn(tmp_sliced_df, drop_name_list)
    # print('After Drop: ', tmp_sliced_df.shape)
    # 填充最后时间离散特征（3个时间段）
    tmp_sliced_df.reset_index(drop=True,inplace=True) ## 一定要reset_index
    setHourColumn(tmp_sliced_df)
    tmp_sliced_df = filledPeriod(tmp_sliced_df)
    # 统计固定点数的片段数
    num_partitions = int(tmp_sliced_df.shape[0] / fixed_point_num)
    # print("划分段数: ", num_partitions, '\n')
    if num_partitions < 1:
        continue
    # 30个点的片段数大于等于1
    for k in range(num_partitions):
        # 第k部分[k*30,(k+1)*30)
        feature_list.append(
            tmp_sliced_df.iloc[k*fixed_point_num:(k+1)*fixed_point_num])
        # if(len(feature_list) == 95):  ###### TEST
        #     print('='*10,"i = ",i," j = ",j)

################################## SAVE ###################################################################
save_num = 9
with open('/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/saved_pickle/cleanTuGong_car'+str(save_num)+'.pkl','wb') as f:
    pickle.dump(feature_list,f)