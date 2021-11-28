#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description : 根据路网进一步细化网格
@Date : 2021/11/28 10:49:59
@Author : gz.gong
@Contact : gz.gong@foxmail.com
@version : 1.0
'''

import numpy as np
from pandas.io import pickle


def path_match(real_roads, car_paths):
    """ assign each catr_path one real road in a single grid

    Arg: 
        real_roads: list[ list[ [float: x, float: y], ... ], ... ]
                    len(real_roads) --> M, M is the number of real road in current grid

        car_paths: list[ list[ [float: x, float: y], ... ], ... ]
                   len(car_paths) --> N, N is the number of car path in current grid
    
    output:
        match_index: list[ int: i_0, int: i_1, ... int: i_N-1 ]
                     len(match_index) --> N
                     i ~ (i_0, i_N-1) ∈ range(0, M)
        
        example: match_index --> list[ int: i_0 = 5, int: i_1 = 2, ... ] NOTES: now M >= 5
                  means car_paths[0] matches real_roads[5],  car_paths[1] matches real_roads[2]

    Description:
        Author: @Tu.xk
        Date: 2021-11-27
    """

    M = len(real_roads)
    N = len(car_paths)
    roadLength_group = [len(real_roads[i]) for i in range(M)]
    roadlength_group = np.array(roadLength_group, dtype=np.int32)
    max_roadLength = np.max(roadlength_group)
    for i in range(M):
        current_roadLen = len(real_roads[i])
        if current_roadLen < max_roadLength:
            for k in range(max_roadLength - current_roadLen):
                real_roads[i].append([0., 0.])
    real_roads_np = np.array(real_roads,
                             dtype=np.float32).reshape(1, M, max_roadLength, 2)

    path_match_index = []
    for j in range(N):
        car_path = car_paths[j]
        car_pathLength = len(car_path)
        car_path = np.array(car_path, dtype=np.float32).reshape(-1, 1, 1, 2)
        real_roads_np_expsion = np.repeat(real_roads_np,
                                          car_pathLength,
                                          axis=0)
        path_match_dis = real_roads_np_expsion - car_path
        path_match_dis = np.linalg.norm(path_match_dis,
                                        ord=2,
                                        axis=-1,
                                        keepdims=False)
        path_match_dis = np.min(path_match_dis, axis=-1)
        path_match_dis = np.sum(path_match_dis, axis=0, keepdims=False)
        index = np.argmin(path_match_dis)
        path_match_index.append(index)

    return path_match_index


from path_slice import Slicer  # FIXME: 需要切分完再调用下面的函数
import pandas as pd
from collections import defaultdict


def readLabel(filenames: list):
    """
    @description : 读取标签文件
    ---------
    @param :
    -------
    @Returns :
    -------
    """
    label_list = []
    for filename in filenames:
        label_dict = np.load(filename, allow_pickle=True).item()
        label_list.append(label_dict)
    return label_list


def tfXYList(data_list: list):
    """
    @description : 将list(dataframe)转成[[[x0,y0],[x1,y1],...],[[],...],...]
    ---------
    @param data_list : list(dataframe)
    -------
    @Returns : [[[x0,y0],[x1,y1],...],[[],...],...]
    -------
    """
    len_df = len(data_list)
    result = []
    for i in range(len_df):
        tmp_data = data_list[i]
        result.append(tmp_data[['x', 'y']].values.tolist())
    return result


def sliceGridByRoad(ft_dict: dict, lb_dict: dict,
                    real_road_dict: dict) -> dict:
    """
    @description  : 按照真实的路进一步细分网格
    ---------
    @param ft_dict : dict, ft_dict[(i,j)]=[pd.DataFrame,...]
    @param lb_dict : dict, lb_dict[(i,j)]=[float,...]
    @param real_road_dict : dict, real_road_dict[(i,j)]=[pd.DataFrame,...]
    -------
    @Returns grid_by_road : dict, grid_by_road[(i,j)]=[energy_of_road1,energy_of_road2,...]
    -------
    """

    xx_range, yy_range = list(ft_dict.keys())[-1]
    result = {}
    for xx in range(xx_range + 1):
        for yy in range(yy_range):
            result[(xx, yy)] = {}
            tmp_lb_list = lb_dict[(xx, yy)]
            if tmp_lb_list == 'none':
                continue
            tmp_data_list = ft_dict[(xx, yy)]
            tmp_road_list = real_road_dict[(xx, yy)]
            tmp_data_list_tf = tfXYList(tmp_data_list)
            tmp_road_list_tf = tfXYList(tmp_road_list)
            match_idx = path_match(
                tmp_road_list_tf,
                tmp_data_list_tf)  # each car match which road

            energy_sum = defaultdict(float)
            mileage_sum = defaultdict(float)
            for data_id, road_id in enumerate(match_idx):
                tmp_data_loop = tmp_data_list[data_id]  # pd.DataFrame
                mileage_tmp = tmp_data_loop['累计里程'].iloc[-1] - tmp_data_loop[
                    '累计里程'].iloc[0]
                energy_tmp = mileage_tmp * tmp_lb_list[data_id]
                if mileage_tmp <= 0:  # 防止小于0
                    mileage_tmp = 0.1
                energy_sum[road_id] += energy_tmp
                mileage_sum[road_id] += mileage_tmp

            for key in energy_sum.keys():
                print(f'key={key}')
                result[(xx, yy)][key] = energy_sum[key] / mileage_sum[key]
    return result


import pickle
# data_feature in /home/Gong.gz/Public_Dataset/ncbdc2021/time_range_avg_heat_map/ft_list.pkl
# data_label in /mnt/PublicDatasets/ncbdc2021/different_time_1127/data{i}_label.npy (i∈range(9))

if __name__ == '__main__':
    time_range_list = [0, 7, 9, 11, 13, 15, 17, 19, 21, 24]  # 提醒我时间范围
    feature_path = '/home/Gong.gz/Public_Dataset/ncbdc2021/time_range_avg_heat_map/ft_list.pkl'
    label_dir = '/mnt/PublicDatasets/ncbdc2021/different_time_1127/'
    label_path_list = [label_dir + f'data{i}_label.npy' for i in range(9)]
    # read feature
    with open(feature_path, 'rb') as f:
        ft_list = pickle.load(f)
    label_list = readLabel(label_path_list)
    # FIXME:加载真实的路
    real_road_dict = None

    grid_by_road = []
    for i_time in range(len(ft_list)):
        grid_by_road.append(
            sliceGridByRoad(ft_list[i_time], label_list[i_time],
                            real_road_dict))
