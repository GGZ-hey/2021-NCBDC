from numpy.lib.npyio import load
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime

car41_path = "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/part-01448-741da358-7624-4bb4-806b-835c106c6b2d.c000.csv"
car29_path = "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/part-00367-741da358-7624-4bb4-806b-835c106c6b2d.c000.csv"
car32_path = "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/part-01749-741da358-7624-4bb4-806b-835c106c6b2d.c000.csv"
car33_path = "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/part-01780-741da358-7624-4bb4-806b-835c106c6b2d.c000.csv"
car39_path = "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/part-00158-741da358-7624-4bb4-806b-835c106c6b2d.c000.csv"
car34_path = "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/part-01002-741da358-7624-4bb4-806b-835c106c6b2d.c000.csv"
car30_path = "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/part-00790-741da358-7624-4bb4-806b-835c106c6b2d.c000.csv"
car62_path = "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/part-01793-741da358-7624-4bb4-806b-835c106c6b2d.c000.csv"
car03_path = "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/part-02889-741da358-7624-4bb4-806b-835c106c6b2d.c000.csv"
car40_path = "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/part-01439-741da358-7624-4bb4-806b-835c106c6b2d.c000.csv"

car_df_original = pd.read_csv(car_path)
car_df_original['经度'] = car_df_original['经度'] / 1e6
car_df_original['维度'] = car_df_original['维度'] / 1e6

# ------------------------------- 经纬度转换 ----------------------------------------------------------
import math


def millerToXY(lon, lat):
    """
    经纬度转换为平面坐标系中的x,y 利用米勒坐标系
    :param lon: 经度
    :param lat: 维度
    :return:
    """
    xy_coordinate = []  # 转换后的XY坐标集
    L = 6381372 * math.pi * 2
    W = L
    H = L / 2
    mill = 2.3
    x = lon * math.pi / 180
    y = lat * math.pi / 180
    y = 1.25 * math.log(math.tan(0.25 * math.pi + 0.4 * y))
    x = (W / 2) + (W / (2 * math.pi)) * x
    y = (H / 2) - (H / (2 * mill)) * y
    xy_coordinate.append((int(round(x)), int(round(y))))
    return xy_coordinate


def millerToLonLat(x, y):
    """
    将平面坐标系中的x,y转换为经纬度，利用米勒坐标系
    :param x: x轴
    :param y: y轴
    :return:
    """
    lonlat_coordinate = []
    L = 6381372 * math.pi * 2
    W = L
    H = L / 2
    mill = 2.3
    lat = ((H / 2 - y) * 2 * mill) / (1.25 * H)
    lat = ((math.atan(math.exp(lat)) - 0.25 * math.pi) * 180) / (0.4 * math.pi)
    lon = (x - W / 2) * 360 / W
    # TODO 最终需要确认经纬度保留小数点后几位
    lonlat_coordinate.append((round(lon, 7), round(lat, 7)))
    return lonlat_coordinate


class DataPreprosser():
    def __init__(self):
        pass

    def dropNan(self, data):
        return data[~((data['车辆状态'].isna()) & (data['充电状态'].isna()) &
                      (data['经度'].isna()))]

    def timeNormalize(self, data):
        return pd.to_datetime(data['时间'], format="%Y-%m-%d %H:%M:%S")

    def calDt(self, data):
        diff_t = data['normalized_time'].diff().copy()
        diff_t.iloc[0] = pd.Timedelta('0 days 00:00:00')
        return diff_t.apply(lambda x: x.seconds)


class ContinuousDriveSlicer():
    def __init__(self, data) -> None:
        self._data = data
        self._result_df = None

    def slicer(self):
        """
        @description  :
        按照
            1. 充电状态（3）
            2. 时间间隔（1h）
        二次分割行驶片段
        ---------
        @param  :
        -------
        @Returns  : List 将行驶片段按照顺序排序，如result[0]是第一个行驶片段的dataframe
        -------
        """
        self._result_df = []
        data = self._data
        # 首先按充电状态划分
        diff_charge_state = (data['充电状态'] == 3).astype(int).diff()
        s_indices = (np.where(diff_charge_state[1:].astype(int) == 1)[0] +
                     1).tolist()
        e_indices = (np.where(diff_charge_state[1:].astype(int) == -1)[0] +
                     1).tolist()
        s_indices.insert(0, 0)
        e_indices.append(diff_charge_state.shape[0])
        assert (s_indices[0] < e_indices[0] and s_indices[-1] < e_indices[-1])
        for i, j in zip(s_indices, e_indices):
            self._result_df.append(data.iloc[i:j])
        # 接着按照时间间隔进一步细化
        i = 0
        while (True):
            # 判断是否处理完成
            if (i == len(self._result_df)):
                break
            sliced_path = self._result_df[i]
            time_threshold = 60 * 60  # 1 hour
            if ((sliced_path['dt'] <=
                 time_threshold).all()):  # 所有小于阈值说明这一段是连续的
                i = i + 1
                continue
            # 细分
            indices = np.where(sliced_path['dt'] > time_threshold)[0].tolist()
            indices.append(sliced_path.shape[0])
            s_time_slice_idx = 0
            for e_time_slice_idx in indices:
                if (s_time_slice_idx == 0):
                    self._result_df[i] = sliced_path.iloc[
                        s_time_slice_idx:e_time_slice_idx]
                else:
                    self._result_df.insert(
                        i, sliced_path.iloc[s_time_slice_idx:e_time_slice_idx])
                i = i + 1
                s_time_slice_idx = e_time_slice_idx

        return self._result_df

    def getSlicedResult(self):
        return self._result_df


# preprocessing
car_df = car_df_original.copy()
pre_processd = DataPreprosser()
car_df['normalized_time'] = pre_processd.timeNormalize(car_df)
car_df = pre_processd.dropNan(car_df)
car_df['dt'] = pre_processd.calDt(car_df)


# TF (lon,lat) -> (x,y)
def tf2miles_x(index):
    xy_coordinate = millerToXY(car_df.iloc[index]['经度'],
                               car_df.iloc[index]['维度'])
    return xy_coordinate[0][0]


def tf2miles_y(index):
    xy_coordinate = millerToXY(car_df.iloc[index]['经度'],
                               car_df.iloc[index]['维度'])
    return xy_coordinate[0][1]


# For Test
print(pd.Series(car_df.index[0:10]).apply(tf2miles_x))
# Reindex
car_df.reset_index(drop=False, inplace=True)
car_df['x'] = pd.Series(car_df.index).apply(tf2miles_x)
car_df['y'] = pd.Series(car_df.index).apply(tf2miles_y)

# slice
slicer = ContinuousDriveSlicer(car_df)
sliced_result = slicer.slicer()

# ---------------------  SAVE  ------------------------------
# import pickle
# save_path = r"/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/saved_pickle/result_NO39_car"
# with open(save_path,'wb') as f:
#     pickle.dump(sliced_result,f)
# ## 读取文件
# with open(save_path, 'rb') as read_f:   #用with的优点是可以不用写关闭文件操作
#     sliced_result_ffile = pickle.load(read_f)
# -----------------------------------------------------------