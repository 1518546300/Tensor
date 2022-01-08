# -*- coding:utf-8 -*-
from typing import ChainMap
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from sklearn.model_selection import train_test_split
import os
import random

class TensorMaxMin():
    def __init__(self, tensor, tensor_mask):
        pos_num = np.where(tensor != 0)

        temp = []
        for i in range(len(pos_num[0])):
            temp.append(tensor[pos_num[0][i]][pos_num[1][i]][pos_num[2][i]])

        self.full_arr = np.array(temp)

        self.min = np.min(self.full_arr)
        self.max = np.max(self.full_arr)
        self.mean = np.mean(self.full_arr)
        self.s = np.std(self.full_arr)
        self.temp_tensor = tensor
        self.tensor_mask = tensor_mask

        #self.ratio = len(self.full_arr) / (x*y*z)
        #0-1归一化
        #self.tensor = (tensor - self.min) / (self.max - self.min) * tensor_mask

    def Max_Min(self):
        #均值归一化
        self.tensor = (self.temp_tensor - self.mean) / (self.max - self.min) * self.tensor_mask

    def Standardization(self):
        self.tensor = (self.temp_tensor - self.mean) / self.s * self.tensor_mask

    # def maxmin_tensor_product(self, tensor, grid):
    #     ret_tensor = np.zeros((self.tensor_1_len, self.tensor_2_len, self.tensor_3_len))
    #     # for t in range(self.tensor_1_len):
    #     #     for i in range(self.tensor_2_len):
    #     #         for j in range(self.tensor_3_len):
    #     #             if grid[i][j] == 1:
    #     #                 ret_tensor[t][i][j] = tensor[t][i][j]
    #     ret_tensor = (tensor - self.min) / (self.max - self.min)
    #     return ret_tensor

    def maxmin_process(self, x):
        _range = self.max - self.min
        return (x - self.min) / _range

    def maxmin_inverse(self, list):
        list = list*(self.max - self.min)
        return list

    def Reconstruct_01(self, tensor):
        return tensor * (self.max - self.min) + self.min

    def Reconstruct_Standardization(self, tensor):
        #return tensor * (self.max) + self.mean
        return tensor * self.s + self.mean

    def Reconstruct_Max_Min(self, tensor):
        #return tensor * (self.max) + self.mean
        return tensor * (self.max - self.min) + self.mean

class China_PM25():
    npz_file_path = "G:\\dataset\\china_hourly\\tensor_data\\"

    def __init__(self, day, split_flag, data_loss_ratio = 0.9):
        self.tensor = self.Data_Read(day)
        self.loss_ratio = data_loss_ratio
        self.Data_Process(split_flag)

    def Data_Read(self, day):
        self.files = os.listdir(self.npz_file_path)
        data = np.load(self.npz_file_path + self.files[day], allow_pickle = True)["tensor_data"]
        return data

    def Data_Process(self, split_flag):
        self.tensor_train = np.full(self.tensor.shape, 0)
        self.tensor_train_mask = np.full(self.tensor.shape, 0)
        self.tensor_test = np.full(self.tensor.shape, 0)
        self.tensor_test_mask = np.full(self.tensor.shape, 0)

        self.Split(split_flag = split_flag)

    def Split(self, split_flag, random_num = 42):
        point_data = []
        
        if split_flag == 0:
            for i in range(len(self.tensor)):
                for j in range(len(self.tensor[i])):
                    for t in range(len(self.tensor[i][j])):
                        point_data.append((i, j, t))
            data_set_len = len(point_data)
            test_num = int(data_set_len * self.loss_ratio)
            train_set, test_set = train_test_split(point_data, test_size = test_num, random_state = random_num)

            for index in train_set:
                self.tensor_train[index[0], index[1], index[2]] = self.tensor[index[0], index[1], index[2]]
                self.tensor_train_mask[index[0], index[1], index[2]] = 1

            for index in test_set:
                self.tensor_test[index[0], index[1], index[2]] = self.tensor[index[0], index[1], index[2]]
                self.tensor_test_mask[index[0], index[1], index[2]] = 1
        elif split_flag == 1:
            for i in range(len(self.tensor)):
                for j in range(len(self.tensor[i])):
                    point_data.append((i, j))
            data_set_len = len(point_data)
            test_num = int(data_set_len * self.loss_ratio)
            train_set, test_set = train_test_split(point_data, test_size = test_num, random_state = random_num)

            for index in train_set:
                self.tensor_train[index[0], index[1], :] = self.tensor[index[0], index[1], :]
                self.tensor_train_mask[index[0], index[1], :] = self.tensor_train_mask[index[0], index[1], :] + 1

            for index in test_set:
                self.tensor_test[index[0], index[1], :] = self.tensor[index[0], index[1], :]
                self.tensor_test_mask[index[0], index[1], :] = self.tensor_test_mask[index[0], index[1], :] + 1
        

def load_map_grid(precision, station_type):
    map_info = np.load('./数据集/地图/' + str(precision) + '_xian_map.npz', allow_pickle=True)['data_grid']
    rows_num = len(map_info)
    cols_num = len(map_info[0])

    table = np.load('./数据集/地图/' + str(precision) + '_precision_map.npz', allow_pickle=True)[station_type]
    
    grid_num = [[0 for i in range(cols_num)] for j in range(rows_num)]
    grid = [[0 for i in range(cols_num)] for j in range(rows_num)]
    for i in range(rows_num):
        for j in range(cols_num):
            if map_info[i][j] != 0:
                grid_num[i][j] = 1
            if table[i][j] != '-2':
                grid[i][j] = 1
    return grid,grid_num

def hour_to_day(year,hour):
    dt = str(year) + '-01-01 0:0:0'
    timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")

    timestamp = time.mktime(timeArray)
    # localtime = time.localtime(timestamp)
    # dt_temp = time.strftime('%Y:%m:%d %H:%M:%S',localtime)

    timestamp = hour*60*60+timestamp
    localtime = time.localtime(timestamp)
    dt_temp = time.strftime('%Y-%m-%d %H:%M:%S',localtime)
    return dt_temp

def split_data_set_2(random_num, grid, test_ratio):
    N = len(grid)
    M = len(grid[0])
    data_set = []

    for i in range(N):
        for j in range(M):
            if grid[i][j] == 1:
                data_set.append([i,j])

    train_num = int(len(data_set)*(1-test_ratio))
    
    train_set, test_set = train_test_split(data_set,test_size = len(data_set)-train_num,random_state = random_num)
    return train_set, test_set

def kriging_set_get(train_set,fusion_set):
    kriging_set = []
    for i in range(len(train_set)):
        kriging_set.append(train_set[i])
    for i in range(len(fusion_set)):
        kriging_set.append(fusion_set[i])
    return kriging_set

def data_set_gain(train_set, test_set, data):
    N = len(data)
    M = len(data[0])
    train_data = [[0 for i in range(M)] for j in range(N)]
    train_grid = [[0 for i in range(M)] for j in range(N)]
    test_data = [[0 for i in range(M)] for j in range(N)]
    test_grid = [[0 for i in range(M)] for j in range(N)]
    for k in range(len(train_set)):
        i_index = train_set[k][0]
        j_index = train_set[k][1]
        if data[i_index][j_index] == -1000:
            train_grid[i_index][j_index] = 0
        else:
            train_data[i_index][j_index] = data[i_index][j_index]
            train_grid[i_index][j_index] = 1

    for k in range(len(test_set)):
        i_index = test_set[k][0]
        j_index = test_set[k][1]
        if data[i_index][j_index] == -1000:
            test_grid[i_index][j_index] = 0
        else:
            test_data[i_index][j_index] = data[i_index][j_index]
            test_grid[i_index][j_index] = 1

    return train_data, train_grid, test_data, test_grid

def load_data(hour_num, data_type, grid, grid_num, precision, station_type):
    rows_num = len(grid)
    cols_num = len(grid[0])

    table = np.load('./数据集/地图/' + str(precision) + '_precision_map.npz', allow_pickle=True)[station_type]

    hour = hour_num
    if station_type == 'shengkong':
        if data_type == "pm2_5":
            data_set = np.load('./数据集/数据/data.npz')['shengkong_pm2_5']
    else:
        if data_type == "pm2_5":
            data_set = np.load('./数据集/数据/data.npz')['xiaoxing_pm2_5']

    data = [[-1 for i in range(cols_num)] for j in range(rows_num)]
    for i in range(rows_num):
        for j in range(cols_num):
            if table[i][j] == "-2":
                num = -1000
            elif grid_num[i][j] == 1:
                sum_num = 0
                station_num = 0
                station_num_array = []
                index_str = table[i][j].split(',')
                for n in range(len(index_str)):
                    index_num = int(table[i][j].split(',')[n])

                    str_num = data_set[hour][index_num]
                    if ~np.isnan(str_num):
                        num = int(str_num)
                        station_num_array.append(num)
                        sum_num = sum_num + num
                        station_num = station_num+1
                if sum_num != 0:
                    num = sum_num/station_num
                    grid[i][j] == 1
                else:
                    num = -1000  
            data[i][j] = num
    return data

def print_test_station(hour_num, data_type, grid, grid_num, precision, station_type, test_grid):
    rows_num = len(test_grid)
    cols_num = len(test_grid[0])

    table = np.load('./数据集/地图/' + str(precision) + '_precision_map.npz', allow_pickle=True)[station_type]

    index_arr_temp = []
    for i in range(rows_num):
        for j in range(cols_num):
            if test_grid[i][j] == 1:
                index_str = table[i][j].split(',')
                for n in range(len(index_str)):
                    index_num = int(table[i][j].split(',')[n])
                    index_arr_temp.append(index_num)
    
    print(index_arr_temp)
    print(len(index_arr_temp))

def day_tensor_data_gain(day_num, data_type, grid, grid_num, train_set, test_set, precision, station_type):
    train_tensor = []
    test_tensor = []
    train_mask_tensor = []
    test_mask_tensor = []
    
    for i in range(day_num*24, (day_num+1)*24):
        data_temp = load_data(i, data_type, grid, grid_num, precision, station_type)
        train_data, train_grid, test_data, test_grid = data_set_gain(train_set, test_set, data_temp)
        #print_test_station(i, data_type, grid, grid_num, precision, station_type, test_grid)
        train_tensor.append(train_data)
        train_mask_tensor.append(train_grid)
        test_tensor.append(test_data)
        test_mask_tensor.append(test_grid)

    train_tensor = np.array(train_tensor)
    train_mask_tensor = np.array(train_mask_tensor)
    test_tensor = np.array(test_tensor)
    test_mask_tensor = np.array(test_mask_tensor)

    return train_tensor, train_mask_tensor, train_grid, test_tensor, test_mask_tensor, test_grid

def data_gain(precision = 1,random_num = 42,day_num = 0, station_type = 'xiaoxing', data_type = "pm2_5"):
    grid, grid_num = load_map_grid(precision, station_type)
    train_set, test_set = split_data_set_2(random_num, grid, 0.3)
    train_tensor_np, train_mask_tensor_np, train_grid, test_tensor_np, test_mask_tensor_np, test_grid = day_tensor_data_gain(day_num, data_type, grid, grid_num, train_set, test_set, precision, station_type)
    return train_tensor_np, train_mask_tensor_np, train_grid, test_tensor_np, test_mask_tensor_np, test_grid

def reproduct_T(train_tensor, train_tensor_mask, test_tensor, test_tensor_mask):
    [k, m, n] = train_tensor.shape
    T = np.zeros((m, n, k))
    T_test = np.zeros((m, n, k))
    T_train_mask = np.zeros((m, n, k))
    T_test_mask = np.zeros((m, n, k))

    for i in range(m):
        for j in range(n):
            T[i , j, :] = train_tensor[:, i, j]
            T_test[i , j, :] = test_tensor[:, i, j]
            T_train_mask[i , j, :] = train_tensor_mask[:, i, j]
            T_test_mask[i , j, :] = test_tensor_mask[:, i, j]
    return T, T_train_mask, T_test, T_test_mask


