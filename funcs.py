import numpy as np
from funcs_kit import data_gain, TensorMaxMin, reproduct_T
from metrics import Tensor_Rmse, Kriging_Tensor_RMSE
from kriging import Kriging_Tensor_Completion
from plot_funcs import grid_map_plot_save

def Kriging_Tensor_Completion_A_Year(start_year, end_year, flag = 0, distance = 10):
    """
    flag == 2 绘制图片
    """
    kriging_rmse_arr = []
    kriging_time_arr = []
    kriging_tensor_arr = []
    for i in range(start_year, end_year):
        train_tensor_np, train_mask_tensor_np, train_grid, test_tensor_np, test_mask_tensor_np, test_grid = data_gain(day_num = i, station_type = 'xiaoxing')

        train_tensor_obj = TensorMaxMin(train_tensor_np, train_mask_tensor_np)
        train_tensor_obj.Max_Min()
        tensor = train_tensor_obj.tensor

        kriging_completion, time = Kriging_Tensor_Completion(i + 1, tensor, train_mask_tensor_np, max_distance = distance)
        new_data = train_tensor_obj.Reconstruct_Max_Min(kriging_completion)

        kriging_rmse, _ = Kriging_Tensor_RMSE(new_data, test_tensor_np, test_mask_tensor_np)
        kriging_rmse_arr.append(kriging_rmse)
        kriging_time_arr.append(time)
        kriging_tensor_arr.append(kriging_completion)
        print('Kriging rmse is: {0};'.format(kriging_rmse))
        print()

        #绘图
        if flag == 2:
            for j in range(len(kriging_completion)):
                grid_map_plot_save(i, j, kriging_completion[j])

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

def Tensor_Plot_Save(day, tensor, file_name):
    for i in range(len(tensor[0][0])):
        grid_map_plot_save(day, i, tensor[:, :, i], file_name)

def Kriging_RMSE():
    file_day_num = [100, 200, 300, 364]
    main_file_name = 'F:\\大气环境\\任务_数据融合\\代码\\tensor\\实验结果\\kriging\\result_'
    kriging_rmse_arr = None
    for i in file_day_num:
        file_name = main_file_name + str(i) + '.npz'
        kriging_data = np.load(file_name, allow_pickle = True)['kriging_rmse_arr']
        if kriging_rmse_arr is None:
            kriging_rmse_arr = kriging_data
        else:
            kriging_rmse_arr = np.hstack((kriging_rmse_arr, kriging_data))
    kriging_rmse = np.mean(kriging_rmse_arr)
    print('2019 Kriging rmse is: {0};'.format(kriging_rmse))
    print()

def Kriging_Plane_Plot(day):
    file_day_num = [100, 200, 300, 364]
    main_file_name = 'F:\\大气环境\\任务_数据融合\\代码\\tensor\\实验结果\\kriging\\result_'
    kriging_plane = None
    for i in file_day_num:
        file_name = main_file_name + str(i) + '.npz'
        kriging_data = np.load(file_name, allow_pickle = True)['kriging_tensor_arr']
        if kriging_plane is None:
            kriging_plane = kriging_data
        else:
            kriging_plane = np.concatenate((kriging_plane, kriging_data), axis = 0)
    data = kriging_plane[day]
    for i in range(len(data)):
        grid_map_plot_save(day, i, data[i])

def SimMat_Construct(X, config_list):
    ndims = len(X.shape)
    shape = X.shape
    ret_mat = np.array([np.identity(shape[i]) for i in range(ndims)])
    start_num = 0.8
    for type in range(3):
        mat_len = len(ret_mat[type])
        int_num = int(config_list[type]/2)

        rate = (1 - start_num) / int_num
        if type != 2:
            for i in range(mat_len):
                k = 0
                for j in range(i - int_num, i + int_num + 1):
                    if (j >= 0 and j < mat_len):
                        if i > j:
                            ret_mat[type][i][j] = start_num + rate * k
                        elif i < j:
                            ret_mat[type][i][j] = 1 - k * rate
                        else:
                            k = 0
                    k = k + 1
        else:
            for i in range(mat_len):
                for j in range(i-1, i+2):
                    if (j >= 0 and j < mat_len):
                        if j == i:
                            ret_mat[type][i][j] = 0
                        else:
                            ret_mat[type][i][j] = 1
            # for j in range(0, mat_len):
            #     if 1 - abs(i-j)/(mat_len/2) > 0:
            #         ret_mat[type][i][j] = 1 - abs(i-j)/(mat_len/2)
            #     else:
            #         ret_mat[type][i][j] = 0
    return ret_mat