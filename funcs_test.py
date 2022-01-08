from re import T
import numpy as np
from pandas.core.indexing import convert_from_missing_indexer_tuple
import os
from scipy.sparse import csgraph
import math
import time
import sys
sys.path.append('F:\\大气环境\\任务_数据融合\\代码\\tensor\\')
from funcs_kit import data_gain, TensorMaxMin, reproduct_T
from funcs import Kriging_RMSE, SimMat_Construct, Kriging_Plane_Plot, Kriging_Tensor_Completion_A_Year, Tensor_Plot_Save
from metrics import Tensor_Rmse, Kriging_Tensor_RMSE
from kriging import Kriging_Tensor_Completion
from plot_funcs import grid_map_plot, tensor_map_plot

def experiment_result_plot():
    filePath = "F:\\大气环境\\任务_数据融合\\代码\\tensor\\实验结果\\AirCP+Smooth+W\\"
    file_list = os.listdir(filePath)
    for i in range(len(file_list)):
        if file_list[i][-4:] == ".npz":
            data = np.load(filePath + file_list[i], allow_pickle = True)

            tensor_data = data["pre_tensor"]
            max_min_tensor = (tensor_data - tensor_data.mean())/(tensor_data.max() - tensor_data.min())

            day_num = file_list[i][59:-4]

            tensor_map_plot(int(day_num), max_min_tensor)

if __name__ == "__main__":
    experiment_result_plot()