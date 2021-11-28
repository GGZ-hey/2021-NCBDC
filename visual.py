from matplotlib.patches import Circle
import numpy as np
import math
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def drawCircle(x0, y0, r):
    '''
    画圆
    x0: 圆心x
    y0: 圆心y
    r: 半径
    '''
    angles_circle = [i * math.pi / 180 for i in range(0, 360)]  # i先转换成double

    x = r * np.cos(angles_circle) + x0
    y = r * np.sin(angles_circle) + y0
    plt.plot(x, y, 'm', linewidth=1)


def drawRect(x, y, long=1000, wide=1000):
    x1 = x - long / 2
    y1 = y + wide / 2

    x2 = x + long / 2
    y2 = y + wide / 2

    x3 = x + long / 2
    y3 = y - wide / 2

    x4 = x - long / 2
    y4 = y - wide / 2

    plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], 'k', linewidth=0.5)


MAX_ENERGY = 10000  # 绘图过程中忽略大于该值的能量值
# time_range_list = list(range(0, 25))
time_range_list = [None, None]  # For season slice
# time_range_list = [0, 7, 9, 11, 13, 15, 17, 19, 21, 24]
for time_id in range(len(time_range_list) - 1):
    average_energy = average_energy_list[time_id]

    # min_x = 33003950.0
    # min_y = 6815930.0
    min_dist = 800  # 距离
    cov = [[(min_dist / 8)**2, 0], [0, (min_dist / 8)**2]]
    interval = 1000
    fig = plt.figure()

    plt.clf()

    # 用于绘制散点图
    xx_scatter = None
    cc_scatter = None
    width, height = list(average_energy.keys())[-1]

    for i in range(width):  # 遍历每一个格子
        for j in range(height):
            if average_energy[(i, j)] == 0:
                continue
            mean = ((2 * i + 1) * interval / 2, (2 * j + 1) * interval / 2)
            # 画点
            # plt.plot(mean[0], mean[1], 'ko', markersize=1)
            # 正态分布采样
            # xx = np.random.multivariate_normal(
            #     mean, cov, int(max(abs(average_energy[(i, j)] / 8), 1)), 'raise')
            if (abs(average_energy[(i, j)]) < MAX_ENERGY):
                xx = np.random.multivariate_normal(
                    mean, cov, int(max(abs(average_energy[(i, j)] / 8), 1)),
                    'raise')
                cc = np.array([0.4 * abs(average_energy[(i, j)])**0.5] *
                              xx.shape[0])
            else:
                xx = np.random.multivariate_normal(mean, cov,
                                                   int(max(abs(1), 1)),
                                                   'raise')
                cc = np.array([0.4 * abs(0)**0.5] * xx.shape[0])
            # 拼接成数组
            if xx_scatter is None:
                xx_scatter = xx
                cc_scatter = cc
            else:
                xx_scatter = np.vstack((xx_scatter, xx))
                cc_scatter = np.concatenate((cc_scatter, cc))

            # 画圆
            # drawCircle(mean[0], mean[1], min_dist / 2)

            # 画矩形
            drawRect(mean[0], mean[1])
            # 把人数显示在图上
            text_str = str(int(average_energy[(i, j)]))
            plt.text(mean[0],
                     mean[1],
                     text_str,
                     fontsize=5,
                     verticalalignment='center',
                     horizontalalignment='center',
                     fontweight='bold')

    # 锚点
    xxyy_no_use = np.array([[-10000, -10000], [-10000, -8000]])
    cc_no_use = np.array([0, 0.4 * abs(MAX_ENERGY)**0.5])
    if xx_scatter is None:
        xx_scatter = xxyy_no_use
        cc_scatter = cc_no_use
    else:
        xx_scatter = np.vstack((xx_scatter, xxyy_no_use))
        cc_scatter = np.concatenate((cc_scatter, cc_no_use))

    # 画散点图
    sc = plt.scatter(xx_scatter[:, 0],
                     xx_scatter[:, 1],
                     s=0.2,
                     alpha=0.4,
                     c=cc_scatter,
                     cmap='coolwarm')
    # plt.colorbar(sc)

    # min_x = 33003950.0
    # min_y = 6815930.0
    # car_num = 6
    # s_idx = 380
    # e_idx = 500
    # plt.plot(loader.data[car_num].iloc[s_idx]['x'],
    #          loader.data[car_num].iloc[s_idx]['y'], 'ko')
    # plt.plot(loader.data[car_num].iloc[e_idx-1]['x'],
    #          loader.data[car_num].iloc[e_idx-1]['y'], 'o',color='y')
    # plt.plot(loader.data[car_num].iloc[s_idx:e_idx]['x'],
    #          loader.data[car_num].iloc[s_idx:e_idx]['y'], 'r')
    # plt.axis('equal')
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    # title_str = f"HeatMap(Ignore>10000) (KJ/KM) {time_range_list[time_id]}:00-{time_range_list[time_id+1]}:00"
    title_str = f"HeatMap(Ignore>10000) (KJ/KM) {sliced_season}"
    plt.title(title_str)
    plt.axis([-1000, 21500, -1000, 12500])
    # plt.savefig(
    #     r'/home/Gong.gz/HDD/my_data/2021-NCBDC/figure/heatmap/season/'
    #     + f"{time_range_list[time_id]}_{time_range_list[time_id+1]}.jpg",
    #     dpi=600)
    plt.savefig(
        r'/home/Gong.gz/HDD/my_data/2021-NCBDC/figure/heatmap/season/'
        + f"{sliced_season}.jpg",
        dpi=600)
