from numpy.lib.npyio import load
from numpy.lib.type_check import real
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime


def _idx(xx):
    return int(np.ceil(xx / 1000) -
               1) if int(np.ceil(xx / 1000) - 1) > 0 else 0


class Slicer:
    def __init__(self, resolution) -> None:
        self.resolution_ = resolution
        self.map_height_ = -1
        self.map_width_ = -1
        self.is_normalized_ = False

    def updateMinMaxXY(self, cars_data, car_num):
        """
        @description : 查找最大最小x、y值
        ---------
        @param cars_data : 汽车数据，格式字典，cars_data[id] -> DataFrame
        @param car_num : 汽车数量
        -------
        @Returns :
        -------
        """
        min_x = 1e9
        max_x = -1e9
        min_y = 1e9
        max_y = -1e9
        for car_id in range(car_num):
            tmp_min_x = cars_data[car_id]['x'].min()
            tmp_min_y = cars_data[car_id]['y'].min()
            tmp_max_x = cars_data[car_id]['x'].max()
            tmp_max_y = cars_data[car_id]['y'].max()
            if tmp_min_x < min_x:
                min_x = tmp_min_x
            if tmp_min_y < min_y:
                min_y = tmp_min_y
            if tmp_max_x > max_x:
                max_x = tmp_max_x
            if tmp_max_y > max_y:
                max_y = tmp_max_y
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.map_height_ = int(np.ceil((max_x - min_x) / self.resolution_) + 1)
        self.map_width_ = int(np.ceil((max_y - min_y) / self.resolution_) + 1)

    def xyNormalize4OneCar(self, data):
        data['x'] = data['x'] - self.min_x
        data['y'] = data['y'] - self.min_y
        return data

    def mapGrid(self, data):
        data['discrete_x'] = data['x'].apply(_idx)
        data['discrete_y'] = data['y'].apply(_idx)
        return data

    def getMapSize(self):
        return (self.map_height_, self.map_width_)

    def setNormalizeFlag(self, is_normalized):
        self.is_normalized_ = is_normalized

    def pathSlice(self, cars_data, car_num, time_range=None):
        """
        @description : 将所有车按格子划分
        ---------
        @param cars_data : 汽车数据，格式字典，cars_data[id] -> DataFrame
        @param car_num : 汽车数量
        -------
        @Returns : 字典，result[(i, j)][car_id][k] -> 第i行第j列格子，car_id车的第k个片段(k从0开始),
                                                                           坐标轴，下x右y
        -------
        """
        self.updateMinMaxXY(cars_data, car_num)
        result = {}
        # initialization
        for i in range(self.map_height_):
            for j in range(self.map_width_):
                result[(i, j)] = {}

        # 遍历
        for car_id in range(car_num):  # 遍历车
            data = cars_data[car_id]
            if (not self.is_normalized_):  # 避免重复归一化
                data = self.xyNormalize4OneCar(data)
            data = self.mapGrid(data)
            if time_range is not None:
                data = data[(data['normalized_time'].apply(lambda x: x.hour) >=
                             time_range[0])
                            & (data['normalized_time'].apply(lambda x: x.hour)
                               < time_range[1])]
            for i in range(self.map_height_):
                for j in range(self.map_width_):
                    # 遍历地图
                    result[(i, j)][car_id] = []
                    if data.empty:
                        continue  # empty data
                    tmp_data_mask = ((data['discrete_x'] == i) &
                                     (data['discrete_y'] == j)).astype(int)
                    # 找到连续序号，并指定起点（格子内起点的上一个点）与终点
                    diff_mask = np.array(tmp_data_mask.diff())
                    start_indices = (
                        np.where(diff_mask[1:].astype(int) == 1)[0] +
                        1).tolist()  # FIXME:考虑前面一个点减小误差
                    end_indices = np.where(
                        diff_mask[1:].astype(int) == -1)[0].tolist()

                    if (len(start_indices) == len(end_indices) + 1):
                        # 最后一段
                        end_indices.append(data.shape[0] - 1)
                    elif (len(start_indices) == len(end_indices) - 1):
                        # 前面一段
                        start_indices.insert(0, 0)
                    else:
                        # 长度一致也有可能包含起点与终点
                        if ((data['discrete_x'].iloc[0] == i) &
                            (data['discrete_y'].iloc[0] == j)):
                            start_indices.insert(0, 0)
                        if ((data['discrete_x'].iloc[-1] == i) &
                            (data['discrete_y'].iloc[-1] == j)):
                            end_indices.append(data.shape[0] - 1)

                    assert (len(start_indices) == len(end_indices))
                    for sid, eid in zip(start_indices, end_indices):
                        # 切片
                        result[(i, j)][car_id].append(data.iloc[sid:eid + 1])

        return result

    def sliceRealRoad(self, real_road: pd.DataFrame, is_normalize_):
        """
        @description : 将真实的路切分到网格
        ---------
        @param real_road : DataFrame with columns[['经度','纬度','x','y']]
        @param is_normalize : if x/y is normalized by zeros(min_x=0, min_y=0) 
        -------
        @Returns : 字典，result[(i, j)][k]:pd.DataFrame -> 第i行第j列格子，第k个真实的路(k从0开始)
        -------
        """
        tmp_real_road = [real_road]
        self.setNormalizeFlag(is_normalize_)
        real_path_slice = self.pathSlice(tmp_real_road, 1, None)
        xx_range, yy_range = list(real_path_slice.keys())[-1]
        # 重新赋值去除内层列表
        for xx in range(xx_range + 1):
            for yy in range(yy_range + 1):
                real_path_slice[(xx, yy)] = real_path_slice[(xx, yy)][0]
        return real_path_slice


class DataLoader:
    def __init__(self, file_names) -> None:
        self.file_names_ = file_names

    def loadOneCar(self, file_name):
        data = pd.read_csv(file_name, low_memory=False)
        return data

    def loadAllData(self):
        self.loaded_data = []
        for file_name in self.file_names_:
            tmp_data = self.loadOneCar(file_name)  # FIXME:去掉nan
            # tmp_data = tmp_data.drop(['Unnamed: 0', '0', 'Unnamed: 0.1'],
            #                          axis=1)
            tmp_data = tmp_data[~(tmp_data['累计里程'].isnull())]
            tmp_data = tmp_data[~(tmp_data['总电压'].isnull())]
            tmp_data = tmp_data[~(tmp_data['总电流'].isnull())]
            tmp_data = tmp_data[~(tmp_data['x'].isnull())]
            tmp_data = tmp_data[(tmp_data['充电状态'] == 2) |
                                (tmp_data['充电状态'] == 3)]
            # TODO:
            tmp_data['normalized_time'] = pd.to_datetime(
                tmp_data['normalized_time'], format="%Y-%m-%d %H:%M:%S")
            self.loaded_data.append(tmp_data)

    def getSeasonData(self, season):
        """
        @description  : 按照季节划分数据
        ---------
        @param  :
            season : 季节str(spring|summer|autumn|winter) 
        -------
        @Returns  : filter data -> List(pd.DataFrame)
        -------
        """
        data_filter = []
        for i in range(len(self.loaded_data)):
            if (season == 'spring'):
                # 3, 4, 5 月
                tmp_data = self.loaded_data[i].copy(deep=True)
                months = tmp_data['normalized_time'].apply(lambda x: x.month)
                tmp_data = tmp_data[(months >= 3) & (months <= 5)]
            elif (season == 'summer'):
                # 6, 7, 8 月
                tmp_data = self.loaded_data[i].copy(deep=True)
                months = tmp_data['normalized_time'].apply(lambda x: x.month)
                tmp_data = tmp_data[(months >= 6) & (months <= 8)]
            elif (season == 'autumn'):
                # 9, 10, 11 月
                tmp_data = self.loaded_data[i].copy(deep=True)
                months = tmp_data['normalized_time'].apply(lambda x: x.month)
                tmp_data = tmp_data[(months >= 9) & (months <= 11)]
            elif (season == 'winter'):
                # 12, 1, 2 月
                tmp_data = self.loaded_data[i].copy(deep=True)
                months = tmp_data['normalized_time'].apply(lambda x: x.month)
                tmp_data = tmp_data[(months >= 12) | ((months >= 1) &
                                                      (months <= 2))]
            else:
                raise ValueError('Season Input Error!')
            tmp_data = tmp_data.reset_index(drop=True)
            data_filter.append(tmp_data)
        return data_filter

    @property
    def data(self):
        return self.loaded_data


class DataSet:
    def __init__(self, result, slicer, car_num) -> None:
        self.label_ = {}
        (height, width) = slicer.getMapSize()
        # initialization
        for i in range(height):
            for j in range(width):
                self.label_[(i, j)] = {}

        for i in range(height):
            for j in range(width):
                for k in range(car_num):
                    self.label_[(i, j)][k] = []
                    for h in range(len(result[(i, j)][k])):
                        result[(i, j)][k][h] = result[(i, j)][k][h].fillna(
                            value=0)  # FIXME:插值填充会更好
                        # FIXME:
                        # 计算每公里能耗
                        dt = pd.to_datetime(
                            result[(i, j)][k][h]['时间'],
                            format="%Y-%m-%d %H:%M:%S").diff().iloc[1:].apply(
                                lambda x: x.seconds)
                        if dt.empty:
                            continue
                        dt[dt >= 10 *
                           60] = 15  # 时间间隔过大（大于10分钟），截断一下，否则会积分出很大的值
                        u_tmp = result[(i, j)][k][h]['总电压']  # 0~1000V
                        # -1000~1000A
                        # i_tmp = np.abs(result[(i, j)][k][h]['总电流']) # FIXME: 直接取绝对值不可行
                        i_tmp = result[(i, j)][k][h]['总电流'].copy()
                        i_tmp[
                            i_tmp < 0] = i_tmp[i_tmp < 0] / 4  # 假设15s有1/4时间在刹车
                        accum_mileage = (
                            result[(i, j)][k][h]['累计里程'].iloc[-1] -
                            result[(i, j)][k][h]['累计里程'].iloc[0]
                        )  # 0~999999.9km
                        if (u_tmp.shape[0] == 1):
                            continue
                        if (accum_mileage == 0):
                            accum_mileage = 0.1
                        assert (accum_mileage != 0)
                        ecpk = np.sum(u_tmp.iloc[:-1].reset_index(drop=True) *
                                      i_tmp.iloc[:-1].reset_index(drop=True) *
                                      dt.reset_index(drop=True)
                                      ) / accum_mileage / 1000  # kJ/km
                        self.label_[(i, j)][k].append(ecpk)

        self.features_ = result

    @property
    def data(self):
        return (self.features_, self.label_)


# --------------------------- MAIN --------------------------------------
car_num = 10
file_names = []
for i in range(car_num):
    file_names.append(
        "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/recleaned_data_tuXkAndGgz/car"
        + str(i) + "fliterDrift.csv")
# for i in range(car_num):
#     file_names.append(
#         "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/drift_preprocessing_csv/car"+str(i+1)+".csv"
#         )
loader = DataLoader(file_names)
loader.loadAllData()
# 按季节划分
is_sliced_byseason = False
sliced_season = 'None'
# 按时间范围划分
time_range_list = [0, 7, 9, 11, 13, 15, 17, 19, 21, 24]
# time_range_list = list(range(0, 25))
# time_range_list = [None,None] # For season slice
ft_list = []
lb_list = []
average_energy_list = []
is_normalized = False
for i in range(len(time_range_list) - 1):  # 遍历时间段
    time_range = [time_range_list[i], time_range_list[i + 1]]
    slicer = Slicer(1000)  # 分辨率1000m
    slicer.setNormalizeFlag(is_normalized)  # 初次运行需要标准化
    # print(time_range)
    if is_sliced_byseason:
        result = slicer.pathSlice(loader.getSeasonData(sliced_season), car_num)
    else:
        result = slicer.pathSlice(loader.data, car_num, time_range)
    is_normalized = True
    data_set = DataSet(result, slicer, car_num)
    (ft, lb) = data_set.data
    ft_list.append(ft)
    lb_list.append(lb)

    # 平均能耗
    average_energy = {}
    for i in range(slicer.getMapSize()[0]):
        for j in range(slicer.getMapSize()[1]):
            cum_mileage = 0.1
            cum_energy = 0
            for k in range(car_num):
                for h in range(len(ft[(i, j)][k])):
                    # 计算每公里能耗
                    dt = pd.to_datetime(
                        ft[(i, j)][k][h]['时间'],
                        format="%Y-%m-%d %H:%M:%S").diff().iloc[1:].apply(
                            lambda x: x.seconds)
                    if dt.empty:
                        continue
                    dt[dt >= 10 * 60] = 15  # 时间间隔过大（大于10分钟），截断一下，否则会积分出很大的值
                    u_tmp = ft[(i, j)][k][h]['总电压']  # 0~1000V
                    # -1000~1000A
                    # i_tmp = np.abs(ft[(i, j)][k][h]['总电流']) # FIXME: 直接取绝对值不可行
                    i_tmp = ft[(i, j)][k][h]['总电流'].copy()
                    i_tmp[i_tmp < 0] = i_tmp[i_tmp < 0] / 4  # 假设15s有1/4时间在刹车
                    accum_mileage = (ft[(i, j)][k][h]['累计里程'].iloc[-1] -
                                     ft[(i, j)][k][h]['累计里程'].iloc[0]
                                     )  # 0~999999.9km
                    if (u_tmp.shape[0] == 1):
                        # print("only one voltage!")
                        continue
                    if accum_mileage < 0:
                        # print('i=',i,'j=',j,'k=',k,'h=',h)
                        # print("Negative accumulated mileage!")
                        continue
                    cum_mileage = cum_mileage + accum_mileage
                    if i == 0 and j == 3:
                        print('NAN!')
                        print('i=', i, 'j=', j, 'k=', k, 'h=', h)
                        print('cum_mileage = ', cum_mileage, '+',
                              accum_mileage)

                    cum_energy = cum_energy + (
                        np.sum(u_tmp.iloc[:-1].reset_index(drop=True) *
                               i_tmp.iloc[:-1].reset_index(drop=True) *
                               dt.reset_index(drop=True)) / 1000)  # kJ/km
            # print('average_energy[(i, j)] = ',cum_energy ,'/',cum_mileage)
            average_energy[(i, j)] = cum_energy / cum_mileage

    average_energy_list.append(average_energy)

# -----------------------------------------------------------------
# 找第四辆车位于x∈(80000,100000),y∈(50000,60000)范围内的路段


def findRoadGivenNum(data, car_id, min_x, max_x, min_y, max_y):
    idx = np.where((data[car_id]['x'] >= min_x)
                   & (data[car_id]['x'] <= max_x)
                   & (data[car_id]['y'] >= min_y)
                   & (data[car_id]['y'] <= max_y))
    return idx


min_x = 33003950.0
min_y = 6815930.0
max_x = 33024548.0
max_y = 6827234.0

idx = findRoadGivenNum(loader.data, 6, min_x + 5000, min_x + 10000,
                       min_y + 4000, min_y + 10000)

# car_4 -> 0:200 indices
# car_5 -> 650:740 indices
#          5500:5736


def calEnergy(data, car_id, start_idx, end_idx):
    tmp_data = data[car_id].iloc[start_idx:end_idx].copy(deep=True)
    dt = pd.to_datetime(
        tmp_data['时间'],
        format="%Y-%m-%d %H:%M:%S").diff().iloc[1:].apply(lambda x: x.seconds)
    if dt.empty:
        return 0
    dt[dt >= 10 * 60] = 15  # 时间间隔过大（大于10分钟），截断一下，否则会积分出很大的值
    u_tmp = tmp_data['总电压']  # 0~1000V
    # -1000~1000A
    # i_tmp = abs(tmp_data['总电流'].copy(deep=True)) ##FIXME: 绝对值计算有问题
    i_tmp = tmp_data['总电流'].copy(deep=True)
    i_tmp[i_tmp < 0] = i_tmp[i_tmp < 0] / 4  # 假设15s有1/4时间在刹车

    energy = np.sum(u_tmp.iloc[:-1].reset_index(drop=True) *
                    i_tmp.iloc[:-1].reset_index(drop=True) *
                    dt.reset_index(drop=True))
    return energy / 1000


# correct_energy2 = calEnergy(loader.data, 5, 0, 30)  # KJ
correct_energy2 = calEnergy(loader.data, 6, 380, 500)  # KJ

# 确保discrete_x/y有值
slicer.updateMinMaxXY(loader.data, car_num)
slicer.min_x = 33003950.0
slicer.min_y = 6815930.0
slicer.max_x = 33024548.0
slicer.max_y = 6827234.0

for car_id in range(6, 7):  # 遍历车
    data = loader.data[car_id]
    data = slicer.xyNormalize4OneCar(data)
    data = slicer.mapGrid(data)


def calEnergyByGrid(data, car_id, start_idx, end_idx, grid):
    tmp_data = data[car_id].iloc[start_idx:end_idx]
    mask = (tmp_data['discrete_x'].diff()
            ) != 0 | (tmp_data['discrete_y'].diff() != 0)
    indices = np.where(mask == True)[0].tolist()
    if indices[-1] != tmp_data.shape[0] - 1:
        indices.append(tmp_data.shape[0] - 1)
    # print(indices)
    cum_energy = 0
    cum_mileage = 0
    for idx in range(len(indices) - 1):
        start_idx, end_idx = indices[idx], indices[idx + 1]
        one_grid_data = tmp_data.iloc[start_idx:end_idx + 1]
        # print(one_grid_data)
        # print(one_grid_data['累计里程'].iloc[-1])
        # print(one_grid_data['累计里程'].iloc[0])
        cum_energy = cum_energy + (one_grid_data['累计里程'].iloc[-1] - one_grid_data['累计里程'].iloc[0]) * \
            grid[(one_grid_data.iloc[0]['discrete_x'],
                  one_grid_data.iloc[0]['discrete_y'])]
        cum_mileage = cum_mileage + \
            (one_grid_data['累计里程'].iloc[-1] - one_grid_data['累计里程'].iloc[0])
        # print((one_grid_data['累计里程'].iloc[-1] - one_grid_data['累计里程'].iloc[0]))
    return cum_energy, cum_mileage


# grid_energy2, cum_km = calEnergyByGrid(loader.data, 5, 0, 30,
#                                        average_energy)
grid_energy2, cum_km = calEnergyByGrid(loader.data, 6, 380, 500,
                                       average_energy)


def calEnergyByGridWithTimeRange(data, car_id, start_idx, end_idx, time_range,
                                 grid_list):
    tmp_data = data[car_id].iloc[start_idx:end_idx]
    mask = (tmp_data['discrete_x'].diff()
            ) != 0 | (tmp_data['discrete_y'].diff() != 0)
    indices = np.where(mask == True)[0].tolist()
    if indices[-1] != tmp_data.shape[0] - 1:
        indices.append(tmp_data.shape[0] - 1)
    # print(indices)
    cum_energy = 0
    cum_mileage = 0
    for idx in range(len(indices) - 1):
        start_idx, end_idx = indices[idx], indices[idx + 1]
        one_grid_data = tmp_data.iloc[start_idx:end_idx + 1]
        # print(one_grid_data)
        # print(one_grid_data['累计里程'].iloc[-1])
        # print(one_grid_data['累计里程'].iloc[0])
        # 找到合适的网格
        which_grid = np.where(
            np.array(time_range) > one_grid_data.iloc[0]
            ['normalized_time'].hour)[0][0] - 1
        grid = grid_list[which_grid]
        cum_energy = cum_energy + (one_grid_data['累计里程'].iloc[-1] - one_grid_data['累计里程'].iloc[0]) * \
            grid[(one_grid_data.iloc[0]['discrete_x'],
                  one_grid_data.iloc[0]['discrete_y'])]
        cum_mileage = cum_mileage + \
            (one_grid_data['累计里程'].iloc[-1] - one_grid_data['累计里程'].iloc[0])
        # print((one_grid_data['累计里程'].iloc[-1] - one_grid_data['累计里程'].iloc[0]))
    return cum_energy, cum_mileage


# grid_energy2, cum_km = calEnergyByGridWithTimeRange(loader.data, 5, 0, 30,
#                                                     time_range_list, average_energy_list)

grid_energy2, cum_km = calEnergyByGridWithTimeRange(loader.data, 6, 380, 500,
                                                    time_range_list,
                                                    average_energy_list)
