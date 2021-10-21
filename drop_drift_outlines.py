from numpy.core.numeric import outer
from numpy.core.numerictypes import maximum_sctype
from numpy.lib.npyio import load
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime


# ----------------- 方形区域去除异常点 --------------------
def filterBySquare(car_df_data):
    x0 = car_df_data['x'].iloc[0]
    y0 = car_df_data['y'].iloc[0]
    # FIXME: 不规范写法
    x_low = -45000 + x0
    x_up = 45000 + x0
    y_low = -20000 + y0
    y_up = 20000 + y0
    car_df_filter = car_df_data[(car_df_data['x'] >= x_low)
                                & (car_df_data['x'] <= x_up) &
                                (car_df_data['y'] >= y_low) &
                                (car_df_data['y'] <= y_up)]
    return car_df_filter


car_df_filter = filterBySquare(car_df)
car_df_filter_state3 = car_df_filter[(car_df_filter['充电状态'] == 3) &
                                     (car_df_filter['车辆状态'] == 1)]  # 删除充电片段

#################### 可视化全局 ###########################
plt.figure(1)
plt.cla()
# plt.scatter(car_df['x']-x0, car_df['y']-y0,marker='o')
plt.scatter(car_df_filter_by_drive_filter['x'] - x0,
            car_df_filter_by_drive_filter['y'] - y0,
            marker='o')
# plt.xlim(-10000,10000)
# plt.ylim(-5000,5000)
plt.savefig('/home/Gong.gz/HDD/2021-NCBDC/scatter_concat_drive_filter2.jpg',
            dpi=300)
##########################################################

# ----------------------------------------------------------------------------------------------------
## 首先需要通过切分后的行驶片段找出s_idx和e_idx
## 接着可视化一下
GPS_ERROR = 20  # m
fig_indices = list(np.arange(1244, 1344))
fig_indices = list(np.arange(134, 144)) + fig_indices

for ii in fig_indices:
    tmp_drive_data = sliced_result[ii].copy(deep=True)
    if tmp_drive_data.empty:
        continue
    tmp_drive_data.reset_index(inplace=True)

    x0 = tmp_drive_data['x'].iloc[0]
    y0 = tmp_drive_data['y'].iloc[0]

    # 首先可视化一下
    plt.figure(2)
    plt.cla()
    X_vec = np.array(tmp_drive_data['x'] - x0).reshape(-1, 1)
    Y_vec = np.array(tmp_drive_data['y'] - y0).reshape(-1, 1)
    n_vec = X_vec.shape[0]
    XY_data = np.concatenate((X_vec, Y_vec), axis=1).reshape(n_vec, 2)
    plt.scatter(XY_data[:, 0], XY_data[:, 1], marker='o')
    plt.savefig('/home/Gong.gz/HDD/2021-NCBDC/figure/scatter' + str(ii) +
                '.jpg',
                dpi=300)

# TODO: 聚类算法
from sklearn.cluster import DBSCAN

############################### 大、小漂移去除 ################################
# 模板匹配前需要去除模板的小漂移
template_idx = 142
xy_template = sliced_result[template_idx]


def filterLittleDrift(df_data, max_velocity=50):
    dx = df_data['x'].diff()
    dx.iloc[0] = 0
    dy = df_data['y'].diff()
    dy.iloc[0] = 0
    dx_square = dx.map(lambda x: x * x)
    dy_square = dy.map(lambda x: x * x)
    distance_square_between = dx_square + dy_square
    dt = df_data['dt'].copy(deep=True)
    dt[dt == 0] = 1  #s
    velocity_between = distance_square_between / dt.map(lambda x: x * x)
    return df_data[velocity_between < max_velocity * max_velocity]


xy_template_filter = filterLittleDrift(xy_template)
xy_template_filter_reindex = xy_template_filter.reset_index(drop=True)

# 模板匹配
outliers_idx = 1332
tested_df = sliced_result[outliers_idx]
min_match_dis = 5000  # m


def findTemplateNotMatchP(tested_df, xy_template, min_match_dis=5000):
    """
    @description  :假设两个df都已经重索引，且(x,y)未初始化
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """

    row_num = tested_df.shape[0]
    row_num_template = xy_template.shape[0]
    print(row_num, row_num_template)
    indices = []
    for i in range(row_num):
        xx = tested_df.iloc[i]['x']
        yy = tested_df.iloc[i]['y']
        xx_se = pd.Series([xx] * row_num_template)
        yy_se = pd.Series([yy] * row_num_template)
        dis_ =  (xx_se - xy_template['x']).map(lambda x:x*x) + \
                   (yy_se - xy_template['y']).map(lambda x:x*x)
        idx_ = dis_.argmin()
        # if (i == 11):
        #     print(tested_df.iloc[i]['时间'])
        #     print(dis_)
        #     print(idx_)
        #     print(dis_[idx_])
        if (dis_.iloc[idx_] >
                min_match_dis * min_match_dis):  # 大于阈值可能为异常点，标记出来
            # print(dis_.iloc[idx_ - 5:idx_ + 5])
            indices.append(i)
    return indices


outliers_idx = findTemplateNotMatchP(tested_df, xy_template_filter_reindex)

# 找出第一个非异常点
first_idx = -1
for i in range(len(outliers_idx)):
    if i != outliers_idx[i]:
        first_idx = i
        break
assert (first_idx + 1 != len(tested_df))


# function
def findFirstNormal(outliers_idx):
    if (len(outliers_idx) == 0):
        return -1

    first_idx = 0
    detect_flag = False
    for i in range(len(outliers_idx)):
        if i != outliers_idx[i]:
            detect_flag = True
            first_idx = i
            break
    if detect_flag == False:
        first_idx = outliers_idx[-1] + 1

    return first_idx


# 异常值填充
tested_df_copy = tested_df.copy(deep=True)
tested_df_copy.reset_index(drop=False, inplace=True)
max_velocity = 50  # m/s
for i in range(0, len(outliers_idx)):  # 默认第一个肯定是对的
    if outliers_idx[i] <= first_idx:
        continue
    if outliers_idx[i] == len(tested_df_copy) - 1:  # 数据最后一个
        print('*' * 10, 'FIRST ↓')
        print(
            tested_df_copy.iloc[outliers_idx[i]][['时间', 'x', 'y', '经度', '维度']])
        tested_df_copy.loc[outliers_idx[i],
                           '经度'] = tested_df_copy.iloc[outliers_idx[i] -
                                                       1]['经度']
        tested_df_copy.loc[outliers_idx[i],
                           '维度'] = tested_df_copy.iloc[outliers_idx[i] -
                                                       1]['维度']
        tested_df_copy.loc[outliers_idx[i],
                           'x'] = tested_df_copy.iloc[outliers_idx[i] - 1]['x']
        tested_df_copy.loc[outliers_idx[i],
                           'y'] = tested_df_copy.iloc[outliers_idx[i] - 1]['y']
        print(
            tested_df_copy.iloc[outliers_idx[i]][['时间', 'x', 'y', '经度', '维度']])
        print('*' * 10, 'END ↑')
    elif i != (len(outliers_idx) - 1) and (outliers_idx[i + 1] -
                                           outliers_idx[i] == 1):
        print('-' * 10, 'FIRST ↓')
        print(
            tested_df_copy.iloc[outliers_idx[i]][['时间', 'x', 'y', '经度', '维度']])
        vx = np.clip((tested_df_copy.iloc[outliers_idx[i] - 1]['x'] -
                      tested_df_copy.iloc[outliers_idx[i] - 2]['x']) /
                     tested_df_copy.iloc[outliers_idx[i] - 1]['dt'],
                     -1 * max_velocity, max_velocity)
        vy = np.clip((tested_df_copy.iloc[outliers_idx[i] - 1]['y'] -
                      tested_df_copy.iloc[outliers_idx[i] - 2]['y']) /
                     tested_df_copy.iloc[outliers_idx[i] - 1]['dt'],
                     -1 * max_velocity, max_velocity)
        tested_df_copy.loc[outliers_idx[i], 'x'] = tested_df_copy.iloc[
            outliers_idx[i] -
            1]['x'] + vx * tested_df_copy.loc[outliers_idx[i], 'dt']
        tested_df_copy.loc[outliers_idx[i], 'y'] = tested_df_copy.iloc[
            outliers_idx[i] -
            1]['y'] + vy * tested_df_copy.loc[outliers_idx[i], 'dt']
        lonlat_coordinate = millerToLonLat(
            tested_df_copy.iloc[outliers_idx[i]]['x'],
            tested_df_copy.iloc[outliers_idx[i]]['y'])
        tested_df_copy.loc[outliers_idx[i], '经度'] = lonlat_coordinate[0][0]
        tested_df_copy.loc[outliers_idx[i], '维度'] = lonlat_coordinate[0][1]
        print(
            tested_df_copy.iloc[outliers_idx[i]][['时间', 'x', 'y', '经度', '维度']])
        print('-' * 10, 'END ↑')
    else:
        print('=' * 10, 'FIRST ↓')
        print(
            tested_df_copy.iloc[outliers_idx[i]][['时间', 'x', 'y', '经度', '维度']])
        tested_df_copy.loc[
            outliers_idx[i],
            'x'] = (tested_df_copy.iloc[outliers_idx[i] - 1]['x'] +
                    tested_df_copy.iloc[outliers_idx[i] + 1]['x']) / 2
        tested_df_copy.loc[
            outliers_idx[i],
            'y'] = (tested_df_copy.iloc[outliers_idx[i] - 1]['y'] +
                    tested_df_copy.iloc[outliers_idx[i] + 1]['y']) / 2
        lonlat_coordinate = millerToLonLat(
            tested_df_copy.iloc[outliers_idx[i]]['x'],
            tested_df_copy.iloc[outliers_idx[i]]['y'])
        tested_df_copy.loc[outliers_idx[i], '经度'] = lonlat_coordinate[0][0]
        tested_df_copy.loc[outliers_idx[i], '维度'] = lonlat_coordinate[0][1]
        print(
            tested_df_copy.iloc[outliers_idx[i]][['时间', 'x', 'y', '经度', '维度']])
        print('=' * 10, 'END ↑')


def paddingDrift(df, outliers_idx, first_idx, repad=False, max_velocity=50):
    # FIXME: 对上面代码改成函数
    tested_df_copy = df.copy(deep=True)
    tested_df_copy.reset_index(drop=True, inplace=True)
    if tested_df_copy.shape[0] == 1:
        return tested_df_copy
    # if (len(outliers_idx) == 1
    #         and outliers_idx[0] >= first_idx + 2):  # 只有一个的情况填充方法
    #     print('a*' * 10, 'FIRST ↓')
    #     # print(tested_df_copy.iloc[outliers_idx[0]][['时间','x','y','经度','维度']])
    #     vx = np.clip((tested_df_copy.iloc[outliers_idx[0] - 1]['x'] -
    #                   tested_df_copy.iloc[outliers_idx[0] - 2]['x']) /
    #                  tested_df_copy.iloc[outliers_idx[0] - 1]['dt'],
    #                  -1 * max_velocity, max_velocity)
    #     vy = np.clip((tested_df_copy.iloc[outliers_idx[0] - 1]['y'] -
    #                   tested_df_copy.iloc[outliers_idx[0] - 2]['y']) /
    #                  tested_df_copy.iloc[outliers_idx[0] - 1]['dt'],
    #                  -1 * max_velocity, max_velocity)
    #     tested_df_copy.loc[outliers_idx[0], 'x'] = tested_df_copy.iloc[
    #         outliers_idx[0] -
    #         1]['x'] + vx * tested_df_copy.loc[outliers_idx[0], 'dt']
    #     tested_df_copy.loc[outliers_idx[0], 'y'] = tested_df_copy.iloc[
    #         outliers_idx[0] -
    #         1]['y'] + vy * tested_df_copy.loc[outliers_idx[0], 'dt']
    #     lonlat_coordinate = millerToLonLat(
    #         tested_df_copy.iloc[outliers_idx[0]]['x'],
    #         tested_df_copy.iloc[outliers_idx[0]]['y'])
    #     tested_df_copy.loc[outliers_idx[0], '经度'] = lonlat_coordinate[0][0]
    #     tested_df_copy.loc[outliers_idx[0], '维度'] = lonlat_coordinate[0][1]
    #     if(outliers_idx[0] == 198):
    #         print(tested_df_copy.iloc[outliers_idx[0] - 1]['x'])
    #         print(tested_df_copy.iloc[outliers_idx[0] - 2]['x'])
    #         print(tested_df_copy.iloc[outliers_idx[0] - 1]['dt'])
    #         print(vx)
    #         print(tested_df_copy.loc[outliers_idx[0], 'x'])
    #     # print(tested_df_copy.iloc[outliers_idx[0]][['时间','x','y','经度','维度']])
    #     # print('*'*10,'END ↑')
    #     return tested_df_copy

    # elif (len(outliers_idx) == 1 and outliers_idx[0] < first_idx + 2):
    #     print('c*' * 10, 'FIRST ↓')
    #     # print(tested_df_copy.iloc[outliers_idx[0]][['时间','x','y','经度','维度']])
    #     tested_df_copy.loc[outliers_idx[0],
    #                        '经度'] = tested_df_copy.iloc[outliers_idx[0] -
    #                                                    1]['经度']
    #     tested_df_copy.loc[outliers_idx[0],
    #                        '维度'] = tested_df_copy.iloc[outliers_idx[0] -
    #                                                    1]['维度']
    #     tested_df_copy.loc[outliers_idx[0],
    #                        'x'] = tested_df_copy.iloc[outliers_idx[0] - 1]['x']
    #     tested_df_copy.loc[outliers_idx[0],
    #                        'y'] = tested_df_copy.iloc[outliers_idx[0] - 1]['y']
    #     # print(tested_df_copy.iloc[outliers_idx[0]][['时间','x','y','经度','维度']])
    #     # print('*'*10,'END ↑')
    #     return tested_df_copy

    for i in range(0, len(outliers_idx)):  # 默认第一个肯定是对的
        if outliers_idx[i] <= first_idx:
            print("love you ", i)
            continue
        if repad or (outliers_idx[i]
                     == len(tested_df_copy) - 1):  # 重新填充 or 数据最后一个
            print('*' * 10, 'FIRST ↓')
            # print(tested_df_copy.iloc[outliers_idx[i]][['时间','x','y','经度','维度']])
            tested_df_copy.loc[outliers_idx[i],
                               '经度'] = tested_df_copy.iloc[outliers_idx[i] -
                                                           1]['经度']
            tested_df_copy.loc[outliers_idx[i],
                               '维度'] = tested_df_copy.iloc[outliers_idx[i] -
                                                           1]['维度']
            tested_df_copy.loc[outliers_idx[i],
                               'x'] = tested_df_copy.iloc[outliers_idx[i] -
                                                          1]['x']
            tested_df_copy.loc[outliers_idx[i],
                               'y'] = tested_df_copy.iloc[outliers_idx[i] -
                                                          1]['y']
            # print(tested_df_copy.iloc[outliers_idx[i]][['时间','x','y','经度','维度']])
            # print('*'*10,'END ↑')
        elif i != (len(outliers_idx) - 1) and (
                outliers_idx[i + 1] - outliers_idx[i] == 1) and (~repad):
            print('-' * 10, 'FIRST ↓')
            # print(tested_df_copy.iloc[outliers_idx[i]][['时间','x','y','经度','维度']])
            vx = np.clip((tested_df_copy.iloc[outliers_idx[i] - 1]['x'] -
                          tested_df_copy.iloc[outliers_idx[i] - 2]['x']) /
                         tested_df_copy.iloc[outliers_idx[i] - 1]['dt'],
                         -1 * max_velocity, max_velocity)
            vy = np.clip((tested_df_copy.iloc[outliers_idx[i] - 1]['y'] -
                          tested_df_copy.iloc[outliers_idx[i] - 2]['y']) /
                         tested_df_copy.iloc[outliers_idx[i] - 1]['dt'],
                         -1 * max_velocity, max_velocity)
            tested_df_copy.loc[outliers_idx[i], 'x'] = tested_df_copy.iloc[
                outliers_idx[i] -
                1]['x'] + vx * tested_df_copy.loc[outliers_idx[i], 'dt']
            tested_df_copy.loc[outliers_idx[i], 'y'] = tested_df_copy.iloc[
                outliers_idx[i] -
                1]['y'] + vy * tested_df_copy.loc[outliers_idx[i], 'dt']
            lonlat_coordinate = millerToLonLat(
                tested_df_copy.iloc[outliers_idx[i]]['x'],
                tested_df_copy.iloc[outliers_idx[i]]['y'])
            tested_df_copy.loc[outliers_idx[i], '经度'] = lonlat_coordinate[0][0]
            tested_df_copy.loc[outliers_idx[i], '维度'] = lonlat_coordinate[0][1]
            # print(tested_df_copy.iloc[outliers_idx[i]][['时间','x','y','经度','维度']])
            # print('-'*10,'END ↑')
        else:
            print('=' * 10, 'FIRST ↓')
            # print(tested_df_copy.iloc[outliers_idx[i]][['时间','x','y','经度','维度']])
            tested_df_copy.loc[
                outliers_idx[i],
                'x'] = (tested_df_copy.iloc[outliers_idx[i] - 1]['x'] +
                        tested_df_copy.iloc[outliers_idx[i] + 1]['x']) / 2
            tested_df_copy.loc[
                outliers_idx[i],
                'y'] = (tested_df_copy.iloc[outliers_idx[i] - 1]['y'] +
                        tested_df_copy.iloc[outliers_idx[i] + 1]['y']) / 2
            lonlat_coordinate = millerToLonLat(
                tested_df_copy.iloc[outliers_idx[i]]['x'],
                tested_df_copy.iloc[outliers_idx[i]]['y'])
            tested_df_copy.loc[outliers_idx[i], '经度'] = lonlat_coordinate[0][0]
            tested_df_copy.loc[outliers_idx[i], '维度'] = lonlat_coordinate[0][1]
            # print(tested_df_copy.iloc[outliers_idx[i]][['时间','x','y','经度','维度']])
            # print('='*10,'END ↑')
    return tested_df_copy


def paddingLittleDrift(df, outliers_idx):
    tested_df_copy = df.copy(deep=True)
    tested_df_copy.reset_index(drop=True, inplace=True)
    outliers_idx = [outliers_idx]
    if (outliers_idx[0] >= 2):  # 只有一个的情况填充方法
        # print('a*' * 10, 'FIRST ↓')
        # print(tested_df_copy.iloc[outliers_idx[0]][['时间','x','y','经度','维度']])
        vx = np.clip((tested_df_copy.iloc[outliers_idx[0] - 1]['x'] -
                      tested_df_copy.iloc[outliers_idx[0] - 2]['x']) /
                     tested_df_copy.iloc[outliers_idx[0] - 1]['dt'],
                     -1 * max_velocity, max_velocity)
        vy = np.clip((tested_df_copy.iloc[outliers_idx[0] - 1]['y'] -
                      tested_df_copy.iloc[outliers_idx[0] - 2]['y']) /
                     tested_df_copy.iloc[outliers_idx[0] - 1]['dt'],
                     -1 * max_velocity, max_velocity)
        tested_df_copy.loc[outliers_idx[0], 'x'] = tested_df_copy.iloc[
            outliers_idx[0] -
            1]['x'] + vx * tested_df_copy.loc[outliers_idx[0], 'dt']
        tested_df_copy.loc[outliers_idx[0], 'y'] = tested_df_copy.iloc[
            outliers_idx[0] -
            1]['y'] + vy * tested_df_copy.loc[outliers_idx[0], 'dt']
        lonlat_coordinate = millerToLonLat(
            tested_df_copy.iloc[outliers_idx[0]]['x'],
            tested_df_copy.iloc[outliers_idx[0]]['y'])
        tested_df_copy.loc[outliers_idx[0], '经度'] = lonlat_coordinate[0][0]
        tested_df_copy.loc[outliers_idx[0], '维度'] = lonlat_coordinate[0][1]
        if (outliers_idx[0] < 10):
            print(tested_df_copy.iloc[outliers_idx[0] - 1]['x'])
            print(tested_df_copy.iloc[outliers_idx[0] - 2]['x'])
            print(tested_df_copy.iloc[outliers_idx[0] - 1]['dt'])
            print(vx)
            print(tested_df_copy.loc[outliers_idx[0], 'x'])
        # print(tested_df_copy.iloc[outliers_idx[0]][['时间','x','y','经度','维度']])
        # print('*'*10,'END ↑')
        return tested_df_copy
    else:
        # print('c*' * 10, 'FIRST ↓')
        # print(tested_df_copy.iloc[outliers_idx[0]][['时间','x','y','经度','维度']])
        tested_df_copy.loc[outliers_idx[0],
                           '经度'] = tested_df_copy.iloc[outliers_idx[0] -
                                                       1]['经度']
        tested_df_copy.loc[outliers_idx[0],
                           '维度'] = tested_df_copy.iloc[outliers_idx[0] -
                                                       1]['维度']
        tested_df_copy.loc[outliers_idx[0],
                           'x'] = tested_df_copy.iloc[outliers_idx[0] - 1]['x']
        tested_df_copy.loc[outliers_idx[0],
                           'y'] = tested_df_copy.iloc[outliers_idx[0] - 1]['y']
        # print(tested_df_copy.iloc[outliers_idx[0]][['时间','x','y','经度','维度']])
        # print('*'*10,'END ↑')
        return tested_df_copy


# 小漂移纠正
little_drift_idx = 1316  # car NO.1
little_drift_df = sliced_result[little_drift_idx]


def calDt(data):
    tt = pd.to_datetime(data['normalized_time'], format="%Y-%m-%d %H:%M:%S")
    diff_t = tt.diff().copy()
    diff_t.iloc[0] = pd.Timedelta('0 days 00:00:00')
    return diff_t.apply(lambda x: x.seconds)


def filterLittleDriftLoop(df_data, max_velocity=50):
    df_data_copy = df_data.copy(deep=True)
    print_f = True
    while (True):
        # 计算小漂移误差
        dx = df_data_copy['x'].diff()
        dx.iloc[0] = 0
        dy = df_data_copy['y'].diff()
        dy.iloc[0] = 0
        dx_square = dx.map(lambda x: x * x)
        dy_square = dy.map(lambda x: x * x)
        distance_square_between = dx_square + dy_square
        dt = calDt(df_data_copy)
        dt[dt == 0] = 1  #s
        velocity_between = distance_square_between / dt.map(lambda x: x * x)
        if (driftJudge(velocity_between, max_velocity)):
            break
        # 修正小漂移误差
        drift_idx = np.where(velocity_between > max_velocity * max_velocity)
        if (print_f):
            print_f = False
            print('drift_idx is ->', drift_idx)
        fixed_idx = drift_idx[0][0]

        n_rows = df_data_copy.shape[0]
        dropped_x = df_data_copy.loc[fixed_idx, 'x']
        dropped_y = df_data_copy.loc[fixed_idx, 'y']
        df_data_copy = df_data_copy.drop(fixed_idx, axis=0)
        fixed_idx += 1
        while ((fixed_idx < n_rows
                )  # and (df_data_copy.loc[fixed_idx, '车速' != 0])
               and (abs(df_data_copy.loc[fixed_idx, 'x'] - dropped_x) < 0.5)
               and (abs(df_data_copy.loc[fixed_idx, 'y'] - dropped_y) < 0.5)):
            # print(fixed_idx)
            df_data_copy = df_data_copy.drop(fixed_idx, axis=0)
            fixed_idx += 1

        df_data_copy.reset_index(drop=True, inplace=True)
        # FIXME: 这一步有待商榷↓
        # df_data_copy = paddingLittleDrift(df_data_copy, fixed_idx)
    return df_data_copy


def driftJudge(velocity_between, max_velocity):
    # print('velocity_between is ->',velocity_between)
    judger = velocity_between > max_velocity * max_velocity
    return velocity_between[judger].empty


little_drift_df_filter = filterLittleDriftLoop(little_drift_df)


def allDriftFilterPerSlice(sliced_df, xy_template):
    """
    @description  : 对单个行驶片段清理漂移异常值
    ---------
    @param sliced_df : 单个行驶片段dataframe
    @param xy_template : 模板
    -------
    @Returns sliced_df_filter : 过滤后的行驶片段df
    -------
    """
    # 大漂移纠正
    outliers_idx = findTemplateNotMatchP(sliced_df, xy_template)
    if (2 * len(outliers_idx) > sliced_df.shape[0]):
        # 异常值太多，不可用
        return sliced_df.iloc[0]
    first_idx = findFirstNormal(outliers_idx)
    if first_idx == -1:
        first_idx = 0
    if first_idx == sliced_df.shape[0]:
        first_idx = sliced_df.shape[0] - 1
    assert (first_idx != len(sliced_df))
    sliced_df_filter = paddingDrift(sliced_df, outliers_idx, first_idx)
    sliced_df_filter = sliced_df_filter.iloc[first_idx:]
    sliced_df_filter.reset_index(drop=True, inplace=True)
    # 小漂移纠正
    sliced_df_filter = filterLittleDriftLoop(sliced_df_filter, 36)
    # 再填充
    outliers_idx = findTemplateNotMatchP(sliced_df_filter, xy_template)
    sliced_df_filter = paddingDrift(sliced_df_filter,
                                    outliers_idx,
                                    0,
                                    repad=True)
    return sliced_df_filter


# 单元测试
little_drift_df_filter = allDriftFilterPerSlice(little_drift_df,
                                                xy_template_filter_reindex)


def allDriftFilter(sliced_result, xy_template):
    """
    @description  : 过滤所有行驶片段
    ---------
    @param sliced_result: list
    @param xy_template: dataframe
    -------
    @Returns sliced_result_filter: list
    -------
    """
    for i in range(len(sliced_result)):
        print("!" * 10, "i = ", i)
        if sliced_result[i].shape[0] <= 10:
            continue
        df_tmp = sliced_result[i].reset_index(drop=True)  # 重索引，扔掉旧索引
        sliced_result[i] = allDriftFilterPerSlice(df_tmp, xy_template)
    return sliced_result


# 这一步之前建议先运行car_df_filter = filterBySquare(car_df)
filter_result = allDriftFilter(sliced_result, xy_template_filter_reindex)

# 直接拼接行驶片段
def concatDriveSlicer(sliced_result):
    car_df_filter_by_drive = None
    update_flag = False
    for i in range(len(sliced_result)):
        if ((sliced_result[i]['index'] > 427189).any()):
            print(i)
        if sliced_result[i].shape[0] < 40:  #10 min
            continue
        if update_flag == False:
            update_flag = True
            car_df_filter_by_drive = sliced_result[i]
        else:
            car_df_filter_by_drive = pd.concat(
                [car_df_filter_by_drive, sliced_result[i]], axis=0)
    return car_df_filter_by_drive


car_df_filter_by_drive = concatDriveSlicer(filter_result)
car_df_filter_by_drive_filter = filterBySquare(car_df_filter_by_drive)

######################### SAVE Pickle ################################
# import pickle
# save_path = "/home/Gong.gz/QunHui-Public_Dataset/Gong.gz/2021-NCBDC/saved_pickle/car1_NO41"
# with open(save_path,'wb') as f:
#     pickle.dump(filter_result,f)
        
# with open(save_path, 'rb') as read_f:   #用with的优点是可以不用写关闭文件操作
#     sliced_result_ffile = pickle.load(read_f)