from numpy.core.numeric import outer
from numpy.core.numerictypes import maximum_sctype
from numpy.lib.npyio import load
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime

# car_df = pd.read_csv('/home/Gong.gz/SSD/2021-NCBDC/car1_df_preprossing.csv')


def setHourColumn(df):
    timee = pd.to_datetime(df['时间'], format="%Y-%m-%d %H:%M:%S")
    hour = timee.apply(lambda x: x.hour)
    df['hour'] = hour


def plotHourBar(car_df, saved_path):
    plt.cla()
    xx = np.array(car_df[car_df['充电状态'] == 3]['hour'].to_list())
    bins = np.arange(0, 24)
    n, bins, patches = plt.hist(xx, bins)
    # plt.show()
    plt.savefig(saved_path, dpi=300)


first_period = [4, 10]
second_period = [10, 16]
third_period = [16, 23]


def filledPeriod(df, first_period=first_period, second_period=second_period, third_period=third_period):
    n_rows = df.shape[0]
    df['dispersed_hour1'] = [0] * n_rows
    df['dispersed_hour2'] = [0] * n_rows
    df['dispersed_hour3'] = [0] * n_rows
    for idx in range(n_rows):
        if df.iloc[idx]['hour'] > first_period[0] and df.iloc[idx][
                'hour'] <= first_period[1]:
            df.loc[idx, 'dispersed_hour1'] = 1
        elif df.iloc[idx]['hour'] > second_period[0] and df.iloc[idx][
                'hour'] <= second_period[1]:
            df.loc[idx, 'dispersed_hour2'] = 1
        elif df.iloc[idx]['hour'] > third_period[0] and df.iloc[idx][
                'hour'] <= third_period[1]:
            df.loc[idx, 'dispersed_hour3'] = 1
    return df
