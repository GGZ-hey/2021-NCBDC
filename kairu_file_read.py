import pickle
import pandas as pd
import numpy as np
from datetime import datetime


# car_1_final = []
# for i in range(len(car_data)):
#     temp = []
#     temp = car_data[i].loc[:, '车速'].values.tolist()
#%%
def time_transfer(former_time, latter_time):
    #输出时间差
    aa = datetime.strptime(former_time, "%Y-%m-%d %H:%M:%S")
    bb = datetime.strptime(latter_time, "%Y-%m-%d %H:%M:%S")
    return (bb - aa).seconds


def time_output(dataframe):
    n = len(dataframe)
    # if n < 2:
    #     time_change = 0
    #     continue
    # if type(dataframe) == pandas.core.series.Series
    if isinstance(dataframe, pd.Series) or n < 2:
        time_change = 0
    else:
        time_change = time_transfer(dataframe.iloc[0]['时间'],
                                    dataframe.iloc[-1]['时间'])
    return time_change


def mile_output(dataframe):
    n = len(dataframe)
    # if n < 2:
    #     mile_change = 0
    if isinstance(dataframe, pd.Series) or n < 2:
        mile_change = 0
    else:
        mile_change = dataframe.iloc[-1]['累计里程'] - dataframe.iloc[0]['累计里程']
    return mile_change


#%%
def shenshao_list(car_data, file_num):
    car1 = {}
    v = []
    voltage = []
    current = []
    time_change = []
    mile = []
    torque = []
    for i in range(len(car_data)):
        v.append(car_data[i]['车速'].tolist())
        voltage.append(car_data[i]['总电压'].tolist())
        current.append(car_data[i]['总电流'].tolist())
        torque.append(car_data[i]['驱动电机转矩'].tolist())
        time_change.append(time_output(car_data[i]))
        mile.append(mile_output(car_data[i]))
    car1['v'] = v
    car1['u'] = voltage
    car1['i'] = current
    car1['T'] = torque
    car1['time'] = time_change
    car1['mile'] = mile
    np.save(
        '/home/Gong.gz/Public_Dataset/ncbdc2021/shenshao_VSP/car{}.npy'
        .format(file_num), car1)


# data = np.load('/home/Wen.kr/car1_NO41.npy', allow_pickle=True).item()
# data = np.load('/home/Wen.kr/ShenShaoData.npy', allow_pickle=True).item()
data = {}
for i in range(10):
    save_path = '/home/Gong.gz/Public_Dataset/ncbdc2021/Data/cleanTuGong_car{}.pkl'.format(
        i)
    with open(save_path, 'rb') as read_f:
        data_temp = pickle.load(read_f)
    print('car{} loaded'.format(i))
    shenshao_list(data_temp, i)
    print('car{} saved'.format(i))

new_data = np.load(
    '/home/Gong.gz/Public_Dataset/ncbdc2021/shenshao_VSP/car1.npy',
    allow_pickle=True).item()
