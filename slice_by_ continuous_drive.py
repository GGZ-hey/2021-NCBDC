from numpy.lib.npyio import load
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime

car_path = "/home/Gong.gz/SSD_DATA/2021-NCBDC/part-00158-741da358-7624-4bb4-806b-835c106c6b2d.c000.csv"
car_df = pd.read_csv(car_path)

class DataPreprosser():
    def __init__(self) -> None:
        pass
    def dropNan(self,data):
            
    def timeNormalize(self,data):
        return pd.to_datetime(data['时间'], format="%Y-%m-%d %H:%M:%S")
    def calDt(self,data):
        pass
    
class ContinuousDriveSlicer():
    def __init__(self,data) -> None:
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

        return self._result_df
    