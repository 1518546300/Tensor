from pykrige.uk import UniversalKriging
import numpy as np
from matplotlib import pyplot as plt
import time
from metrics import Tensor_Rmse, Tensor_Rmse
from kriging import Kriging_Tensor_Completion
import sys
from plot_funcs import grid_map_plot, grid_map_plot_save
sys.path.append('F:\\大气环境\\任务_数据融合\\代码\\tensor\\')
from funcs_kit import data_gain, TensorMaxMin, reproduct_T, China_PM25
from metrics import Tensor_Rmse

def krige_interplotation():

    # 已知采样点的数据，是坐标（x，y）和坐标对应的值
    # 矩阵中第一列是x,第二列是y,第三列是坐标对应的值
    data = np.array(
        [
            [0.1, 0.1, 0.9],
            [0.2, 0.1, 0.8],
            [0.1, 0.3, 0.9],
            [0.5, 0.4, 0.5],
            [0.3, 0.3, 0.7],
        ])

    # 网格
    x_range = 0.6
    y_range = 0.6
    range_step = 0.1
    gridx = np.arange(0.0, x_range, range_step) #三个参数的意思：范围0.0 - 0.6 ，每隔0.1划分一个网格
    gridy = np.arange(0.0, y_range, range_step)

    ok3d = u(data[:, 0], data[:, 1], data[:, 2], variogram_model="linear") # 模型
    # variogram_model是变差函数模型，pykrige提供 linear, power, gaussian, spherical, exponential, hole-effect几种variogram_model可供选择，默认的为linear模型。
    # 使用不同的variogram_model，预测效果是不一样的，应该针对自己的任务选择合适的variogram_model。

    k3d1, ss3d = ok3d.execute("grid", gridx, gridy) # k3d1是结果，给出了每个网格点处对应的值

    print(np.round(k3d1,2))

    # 绘图
    fig, (ax1) = plt.subplots(1)
    ax1.imshow(k3d1, origin="lower")
    ax1.set_title("ordinary krige")
    plt.tight_layout()
    plt.show()

def data_process(grid_data, mask):
    [x, y] = grid_data.shape
    ret_arr = []
    for i in range(x):
        for j in range(y):
            if mask[i][j] == 1:
                ret_arr.append([i, j, grid_data[i][j]])
    return np.array(ret_arr)

def Kriging_Tensor_Completion(day, train_tensor, train_grid, max_distance = 10, variogram_model = "linear"):
    # 网格
    [x, y, z] = train_tensor.shape

    x_range = x
    y_range = y
    range_step = 1
    gridx = np.arange(0.0, x_range, range_step) #三个参数的意思：范围0.0 - 0.6 ，每隔0.1划分一个网格
    gridy = np.arange(0.0, y_range, range_step)

    print()
    print('************************ Day: {0} Krige **************************'.format(day))
    start = time.time()
    completion_tensor = train_tensor.copy()
    
    for t in range(z):
        data = data_process(train_tensor[:, :, t], train_grid[:, :, t])
        ok3d = UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model = variogram_model) # 模型
        # variogram_model是变差函数模型，pykrige提供 linear, power, gaussian, spherical, exponential, hole-effect几种variogram_model可供选择，默认的为linear模型。
        # 使用不同的variogram_model，预测效果是不一样的，应该针对自己的任务选择合适的variogram_model。

        k3d1, ss3d = ok3d.execute("grid", gridx, gridy) # k3d1是结果，给出了每个网格点处对应的值
        temp_arr = np.round(k3d1,2)
        completion_tensor[:, :, t] = temp_arr.T
    end = time.time()
    print('Krige running time: %d seconds'%(end - start))
    print('*********************************************************')
    print()
    return completion_tensor, end - start

def Kriging_Tensor_Completion_A_Year(start_day, end_day, data_flag, split_flag, ratio_num = 0.9, flag = 0, distance = 10):
    """
    flag == 2 绘制图片
    """
    kriging_rmse_arr = []
    kriging_time_arr = []
    kriging_tensor_arr = []
    main_tensor = []
    variogram_model = "linear"
    for i in range(start_day, end_day):
        if data_flag == 0:
            train_tensor_np, train_mask_tensor_np, train_grid, test_tensor_np, test_mask_tensor_np, test_grid = data_gain(day_num = i, station_type = 'xiaoxing')
            train_tensor_np, train_mask_tensor_np, test_tensor_np, test_mask_tensor_np = reproduct_T(train_tensor_np, train_mask_tensor_np, test_tensor_np, test_mask_tensor_np)
        elif data_flag == 1:
            china_data = China_PM25(i, split_flag, ratio_num)
            train_tensor_np = china_data.tensor_train
            train_mask_tensor_np = china_data.tensor_train_mask
            test_tensor_np = china_data.tensor_test
            test_mask_tensor_np = china_data.tensor_test_mask

        train_tensor_obj = TensorMaxMin(train_tensor_np, train_mask_tensor_np)
        train_tensor_obj.Max_Min()
        tensor = train_tensor_obj.tensor

        kriging_completion, time = Kriging_Tensor_Completion(i + 1, tensor, train_mask_tensor_np, max_distance = distance, variogram_model = variogram_model)
        new_data = train_tensor_obj.Reconstruct_Max_Min(kriging_completion)

        kriging_rmse, _ = Tensor_Rmse(new_data, test_tensor_np, test_mask_tensor_np)
        main_tensor.append(new_data)
        kriging_rmse_arr.append(kriging_rmse)
        kriging_time_arr.append(time)
        kriging_tensor_arr.append(kriging_completion)
        print('Kriging rmse is: {0};'.format(kriging_rmse))
        print()

        #绘图 flag = 1 单张绘制，flag = 2 张量图
        if flag == 1:
            for j in range(len(kriging_completion)):
                grid_map_plot(i, j, kriging_completion[j])
        elif flag == 2:
            for j in range(len(kriging_completion)):
                grid_map_plot_save(i, j, kriging_completion[j], 'krige_plot')

        # if flag == 1 and ((i > 0 and i % 100 == 0) or (i == 364)):
        #     np.savez('F:\\大气环境\\任务_数据融合\\代码\\tensor\\实验结果\\kriging\\result_' + str(i) + '.npz',
        #         test_mask_tensor = test_mask_tensor_np,
        #         kriging_rmse_arr = kriging_rmse_arr,
        #         kriging_time_arr = kriging_time_arr,
        #         kriging_tensor_arr = kriging_tensor_arr
        #     )
        #     kriging_rmse_arr = []
        #     kriging_time_arr = []
        #     kriging_tensor_arr = []
    print('************** 2019 **************')
    print()
    main_rmse = np.array(kriging_rmse_arr).mean()
    print('2019 main rmse is : %f '%(main_rmse))
    print()
    print('**********************************')
    print()

    if start_day == 0 and end_day == 365:
        np.savez('F:\\大气环境\\任务_数据融合\\代码\\tensor\\实验结果\\krige\\result_' + str(end_day - start_day) + '_' + str(variogram_model) + '.npz',
                pre_rmse = kriging_rmse_arr,
                pre_data = main_tensor
        )

if __name__ == '__main__':
    # data_flag = 0 西安数据，data_flag = 1 中国数据
    data_flag = 1

    start_day = 0
    end_day = 2
    ratio_num = 0.99
    split_flag = 1

    Kriging_Tensor_Completion_A_Year(start_day, end_day, split_flag = split_flag, flag = 0, data_flag = data_flag, ratio_num = ratio_num)