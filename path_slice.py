from numpy.lib.npyio import load
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime


def _idx(xx):
    return int(np.ceil(xx/1000)-1) if int(np.ceil(xx/1000)-1) > 0 else 0


class Slicer:
    def __init__(self, resolution) -> None:
        self.resolution_ = resolution
        self.map_height_ = -1
        self.map_width_ = -1

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
        self.map_height_ = int(np.ceil((max_x - min_x)/self.resolution_) + 1)
        self.map_width_ = int(np.ceil((max_y - min_y)/self.resolution_) + 1)

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

    def pathSlice(self, cars_data, car_num):
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
            data = self.xyNormalize4OneCar(data)
            data = self.mapGrid(data)
            for i in range(self.map_height_):
                for j in range(self.map_width_):
                    # 遍历地图
                    result[(i, j)][car_id] = []
                    tmp_data_mask = ((data['discrete_x'] == i) &
                                     (data['discrete_y'] == j)).astype(int)
                    # 找到连续序号，并指定起点（格子内起点的上一个点）与终点
                    diff_mask = np.array(tmp_data_mask.diff())
                    start_indices = (np.where(diff_mask[1:].astype(int) == 1)[
                        0]+1).tolist()  # FIXME:考虑前面一个点减小误差
                    end_indices = np.where(
                        diff_mask[1:].astype(int) == -1)[0].tolist()

                    if(len(start_indices) == len(end_indices) + 1):
                        # 最后一段
                        end_indices.append(data.shape[0] - 1)
                    elif(len(start_indices) == len(end_indices) - 1):
                        # 前面一段
                        start_indices.insert(0, 0)
                    else:
                        # 长度一致也有可能包含起点与终点
                        if ((data['discrete_x'].iloc[0] == i) & (data['discrete_y'].iloc[0] == j)):
                            start_indices.insert(0, 0)
                        if ((data['discrete_x'].iloc[-1] == i) & (data['discrete_y'].iloc[-1] == j)):
                            end_indices.append(data.shape[0] - 1)

                    assert(len(start_indices) == len(end_indices))
                    for sid, eid in zip(start_indices, end_indices):
                        # 切片
                        result[(i, j)][car_id].append(data.iloc[sid:eid + 1])

        return result


class DataLoader:
    def __init__(self, file_names) -> None:
        self.file_names_ = file_names

    def loadOneCar(self, file_name):
        data = pd.read_csv(file_name)
        return data

    def loadAllData(self):
        self.loaded_data = []
        for file_name in self.file_names_:
            tmp_data = self.loadOneCar(file_name)  # FIXME:去掉nan
            tmp_data = tmp_data[~(tmp_data['x'].isnull())]
            tmp_data = tmp_data[(tmp_data['充电状态'] == 2) |
                                (tmp_data['充电状态'] == 3)]
            self.loaded_data.append(tmp_data)

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
                        result[(i, j)][k][h].fillna(value=0)  # FIXME:插值填充会更好
                        # FIXME:
                        # 计算每公里能耗
                        dt = pd.to_datetime(result[(
                            i, j)][k][h]['时间'], format="%Y-%m-%d %H:%M:%S").diff().iloc[1:].apply(lambda x: x.seconds)
                        u_tmp = result[(i, j)][k][h]['总电压']  # 0~1000V
                        # -1000~1000A
                        i_tmp = np.abs(result[(i, j)][k][h]['总电流'])
                        accum_mileage = (result[(i, j)][k][h]['累计里程'].iloc[-1] -
                                         result[(i, j)][k][h]['累计里程'].iloc[0])  # 0~999999.9km
                        if(u_tmp.shape[0] == 1):
                            continue
                        if (accum_mileage == 0):
                            accum_mileage = 0.1
                        assert(accum_mileage != 0)
                        ecpk = np.sum(
                            u_tmp.iloc[:-1]*i_tmp.iloc[:-1]*dt) / accum_mileage / 1000  # kJ/km
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
        "/home/Gong.gz/GGZ_grade2/2021-NCBDC/car_csv_new/car_csv/car_" + str(i+1) + ".csv")
loader = DataLoader(file_names)
loader.loadAllData()
slicer = Slicer(1000)
result = slicer.pathSlice(loader.data, car_num)
data_set = DataSet(result, slicer, car_num)
(ft, lb) = data_set.data

# 平均能耗
average_energy = {}
for i in range(slicer.getMapSize()[0]):
    for j in range(slicer.getMapSize()[1]):
        cum_mileage = 0.1
        cum_energy = 0
        for k in range(car_num):
            for h in range(len(ft[(i, j)][k])):
                # 计算每公里能耗
                dt = pd.to_datetime(ft[(
                    i, j)][k][h]['时间'], format="%Y-%m-%d %H:%M:%S").diff().iloc[1:].apply(lambda x: x.seconds)
                u_tmp = ft[(i, j)][k][h]['总电压']  # 0~1000V
                # -1000~1000A
                i_tmp = np.abs(ft[(i, j)][k][h]['总电流'])
                accum_mileage = (ft[(i, j)][k][h]['累计里程'].iloc[-1] -
                                 ft[(i, j)][k][h]['累计里程'].iloc[0])  # 0~999999.9km
                if(u_tmp.shape[0] == 1):
                    print("cont")
                    continue
                cum_mileage = cum_mileage + accum_mileage
                cum_energy = cum_energy + (np.sum(
                    u_tmp.iloc[:-1]*i_tmp.iloc[:-1]*dt) / 1000)  # kJ/km
        average_energy[(i, j)] = cum_energy / cum_mileage

# -----------------------------------------------------------------
# 找第四辆车位于x∈(80000,100000),y∈(50000,60000)范围内的路段


def findRoadGivenNum(data, car_id, min_x, max_x, min_y, max_y):
    idx = np.where((data[car_id]['x'] >= min_x)
                   & (data[car_id]['x'] <= max_x)
                   & (data[car_id]['y'] >= min_y)
                   & (data[car_id]['y'] <= max_y)
                   )
    return idx


idx = findRoadGivenNum(loader.data, 4, 80000, 100000, 50000, 60000)

# car_4 -> 0:200 indices
# car_5 -> 650:740 indices
#          5500:5736


def calEnergy(data, car_id, start_idx, end_idx):
    tmp_data = data[car_id].iloc[start_idx: end_idx]
    dt = pd.to_datetime(
        tmp_data['时间'], format="%Y-%m-%d %H:%M:%S").diff().iloc[1:].apply(lambda x: x.seconds)
    u_tmp = tmp_data['总电压']  # 0~1000V
    # -1000~1000A
    i_tmp = abs(tmp_data['总电流'])
    energy = np.sum(u_tmp.iloc[:-1]*i_tmp.iloc[:-1]*dt)
    return energy / 1000


correct_energy2 = calEnergy(loader.data, 5, 5500, 5736)  # KJ


def calEnergyByGrid(data, car_id, start_idx, end_idx, grid):
    tmp_data = data[car_id].iloc[start_idx: end_idx]
    mask = (tmp_data['discrete_x'].diff()) != 0 | (
        tmp_data['discrete_y'].diff() != 0)
    indices = np.where(mask == True)[0]
    # print(indices)
    cum_energy = 0
    cum_mileage = 0
    for idx in range(len(indices.tolist())-1):
        start_idx, end_idx = indices[idx], indices[idx+1]
        one_grid_data = tmp_data.iloc[start_idx:end_idx]
        # print(one_grid_data)
        # print(one_grid_data['累计里程'].iloc[-1])
        # print(one_grid_data['累计里程'].iloc[0])
        cum_energy = cum_energy + (one_grid_data['累计里程'].iloc[-1] - one_grid_data['累计里程'].iloc[0]) * \
            grid[(one_grid_data.iloc[0]['discrete_x'],
                  one_grid_data.iloc[0]['discrete_y'])]
        cum_mileage = cum_mileage + \
            (one_grid_data['累计里程'].iloc[-1] - one_grid_data['累计里程'].iloc[0])
    return cum_energy, cum_mileage


grid_energy2, cum_km = calEnergyByGrid(
    loader.data,  5, 5500, 5736, average_energy)
