from numpy.lib.npyio import load
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime

car_path = "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/part-00158-741da358-7624-4bb4-806b-835c106c6b2d.c000.csv"
car_df_original = pd.read_csv(car_path)


class DataPreprosser():
    def __init__(self):
        pass

    def dropNan(self, data):
        return data[~((data['车辆状态'].isna()) & (data['充电状态'].isna()) &
                      (data['经度'].isna()))]

    def timeNormalize(self, data):
        return pd.to_datetime(data['时间'], format="%Y-%m-%d %H:%M:%S")

    def calDt(self, data):
        diff_t = data['normalized_time'].diff()
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
            self._result_df.append(data.iloc[i, j])
        # 接着按照时间间隔进一步细化
        i = 0
        while (True):
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
                    self._result_df[i] = data.iloc[
                        s_time_slice_idx:e_time_slice_idx]
                else:
                    self._result_df.insert(
                        i, data.iloc[s_time_slice_idx:e_time_slice_idx])
                i = i + 1
                s_time_slice_idx = e_time_slice_idx
            # 判断是否处理完成
            if (i == len(self._result_df)):
                break

        return self._result_df

    def getSlicedResult(self):
        return self._result_df


# preprocessing
car_df = car_df_original
pre_processd = DataPreprosser()
car_df['normalized_time'] = pre_processd.timeNormalize(car_df)
car_df = pre_processd.dropNan(car_df)
car_df['dt'] = pre_processd.calDt(car_df)
# slice
slicer = ContinuousDriveSlicer(car_df)
sliced_result = slicer.slicer()