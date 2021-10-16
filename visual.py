import numpy as np
import math
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import Circle

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


def drawRect(x, y, long=1002, wide=679):
	x1 = x - long / 2
	y1 = y + wide / 2

	x2 = x + long / 2
	y2 = y + wide / 2

	x3 = x + long / 2
	y3 = y - wide / 2

	x4 = x - long / 2
	y4 = y - wide / 2

	plt.plot([x1, x2, x3, x4, x1], [y1, y2, y3, y4, y1], 'k', linewidth=0.5)


reclassify_num = 26187
min_dist = 679.0  # 距离
cov = [[(min_dist / 8)**2, 0], [0, (min_dist / 8)**2]]
fig = plt.figure()
save_num = 1
for i in range(reclassify_num):  # 遍历所有时间点
	plt.clf()
	fig = plt.figure()
 
	# 用于绘制散点图
	xx_scatter = None
	cc_scatter = None
 
	data = reclassify_by_date[i]
 
	for j in range(data.shape[0]):  # 遍历每一个格子
		mean = (data.iloc[j]['x'], data.iloc[j]['y'])
		# 画点
		plt.plot(mean[0], mean[1], 'ro', markersize=2)
		# 正态分布采样
		xx = np.random.multivariate_normal(mean, cov, (data.iloc[j]['cnt']),
										   'raise')
		cc = np.array([data.iloc[j]['cnt']] * xx.shape[0])
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
		text_str = 'id: ' + str(data.iloc[j]['reindex_id']) + '\n' + str(
			data.iloc[j]['cnt'])
		plt.text(mean[0],
				 mean[1],
				 text_str,
				 fontsize=5,
				 verticalalignment='center',
				 horizontalalignment='center',
				 fontweight='bold')
	# 画散点图
	plt.scatter(xx_scatter[:, 0],
				xx_scatter[:, 1],
				s=0.1,
				alpha=0.6,
				c=cc_scatter,
				cmap='coolwarm')

	# plt.axis('equal')
	plt.xlabel("x/m")
	plt.ylabel("y/m")
	date_str = data['datatime_format'].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
	plt.title("date: " + date_str)
	plt.savefig(r'/home/Gong.gz/HHD_DATA/2021-Zhuhai/figure/' + str(save_num) +
				'.jpg',
				dpi=150)
	save_num = save_num + 1