from datetime import time
import pickle
import numpy as np
import pandas as pd
import copy

with open(
        "/home/Gong.gz/Public_Dataset/ncbdc2021/time_range_avg_heat_map/avg_list_24_hour.pkl",
        "rb") as f:
    average_energy_list = pickle.load(f)

i_range = 22
j_range = 13

have_data = {}

# initialization
for i in range(i_range):
    for j in range(j_range):
        have_data[(i, j)] = False

# set True if (i,j) have data
for i in range(i_range):
    for j in range(j_range):
        for time_id in range(len(average_energy_list)):
            if (average_energy_list[time_id][(i, j)] != 0):
                have_data[(i, j)] = True
                break


def getSearchList(k, max_length):
    """
    @description  : 保证查找由最近点开始
    例如 : 
    data_list = [1,3,5,4,2]，假设k=2,即data_list[k]=5
    查找顺序 5->3->4->1->2
    ---------
    @param  :
    -------
    @Returns  : 查找顺序索引list
    -------
    """
    assert(k<max_length),'param @1 must less than param @2!'
    step = 1
    step_with_neg = step
    k_step = k
    result = []
    for i in range(max_length - 1):
        step_with_neg = -1 * np.sign(step_with_neg) * step
        k_step += step_with_neg
        if k_step < 0 or k_step >= max_length:
            break
        result.append(k_step)
        step += 1
        
    if k_step < 0:
        step_with_neg = -1 * np.sign(step_with_neg) * (step + 1)
        assert (k_step + step_with_neg <
                max_length), 'wrong : k_step + step_with_neg >= max_length!'
        k_step += step_with_neg
        while (k_step < max_length):
            result.append(k_step)
            k_step += 1
    elif k_step >= max_length:
        step_with_neg = -1 * np.sign(step_with_neg) * (step + 1)
        assert (k_step + step_with_neg >=
                0), 'wrong : k_step + step_with_neg < 0!'
        k_step += step_with_neg
        while (k_step >= 0):
            result.append(k_step)
            k_step -= 1
    return result


average_energy_list_filled = copy.deepcopy(average_energy_list)

for i in range(i_range):
    for j in range(j_range):
        if (not have_data[(i, j)]):
            continue
        # have data -> filled if order
        for k in range(len(average_energy_list)):
            if (average_energy_list[k][(i, j)] != 0):
                continue
            search_idxs = getSearchList(k,len(average_energy_list))
            for search_id in search_idxs:
                if average_energy_list[search_id][(i,j)] == 0:
                    continue
                else:
                    average_energy_list_filled[k][(i,j)] = average_energy_list[search_id][(i,j)]
                    break
                
################################### SAVE ###################################
with open("/home/Gong.gz/Public_Dataset/ncbdc2021/time_range_avg_heat_map/avg_list_24_hour_filled.pkl",'wb') as f:
    pickle.dump(average_energy_list_filled,f)