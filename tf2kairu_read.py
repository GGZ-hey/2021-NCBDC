#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description : 将ft_list转化为VSP读取文件的格式
@Date : 2021/11/24 23:46:16
@Author : gz.gong
@Contact : gz.gong@foxmail.com
@version : 1.0
'''


def tf2krReadByDict(ft_list: list, car_num: int) -> list:
    x_range, y_range = list(ft_list[0].keys())[-1]
    result = []
    for i_time_range in range(len(ft_list)):
        result.append({})
        for xx in range(x_range + 1):
            for yy in range(y_range + 1):
                result[i_time_range][(xx, yy)] = []
                for car_id in range(car_num):
                    for sliced_id in range(
                            len(ft_list[i_time_range][(xx, yy)][car_id])):
                        data_df_tmp = ft_list[i_time_range][(
                            xx, yy)][car_id][sliced_id]
                        if (data_df_tmp.shape[0] > 8):  # 大于5个点才要
                            result[i_time_range][(xx, yy)].append(data_df_tmp)
    return result


def printResult(result_ft_list):
    x_range, y_range = list(result_ft_list[0].keys())[-1]
    for i_time_range in range(len(result_ft_list)):
        for xx in range(x_range + 1):
            for yy in range(y_range + 1):
                print(f"len(result_ft_list[{i_time_range}][({xx},{yy})])" +
                      str(len(result_ft_list[i_time_range][(xx, yy)])))
