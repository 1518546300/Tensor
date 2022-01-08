import math
import numpy as np

def Grid_Rmse(x,y,grid):
    rmse_list = []
    N = len(grid)
    M = len(grid[0])
    for i in range(N):
        for j in range(M):
            if grid[i][j] == 1:
                x1 = x[i][j]
                x2 = y[i][j]
                rmse_list.append((x1-x2)*(x1-x2))
    rmse = math.sqrt(np.mean(rmse_list))
    return rmse

def Rmse(x,y):
    rmse_list = []
    for i in range(len(x)):
        x1 = x[i]
        x2 = y[i]
        rmse_list.append((x1-x2)*(x1-x2))
    rmse = math.sqrt(np.mean(rmse_list))
    return rmse

def Mae(x,y):
    mae_list = []
    for i in range(len(x)):
        x1 = x[i]
        x2 = y[i]
        mae_list.append(abs(x1-x2))
    ret_mae = np.mean(mae_list)
    return ret_mae

def Relative_error(x,y):
    relative_error_list = []
    for i in range(len(x)):
        x1 = x[i]
        x2 = y[i]
        relative_error_list.append(abs(x1-x2)/x1*100)
    relative_error = sum(relative_error_list)/len(relative_error_list)
    return relative_error

def Tensor_Rmse(completion_tensor, test_tensor, tensor_grid):
    pre_tensor = completion_tensor * tensor_grid
    [x, y, z] = pre_tensor.shape
    
    rmse_list = []

    for t in range(z):
        temp_list = []
        for i in range(x):
            for j in range(y):
                if tensor_grid[i][j][t] == 1:
                    x1 = pre_tensor[i][j][t]
                    x2 = test_tensor[i][j][t]
                    temp_num = (x1 - x2) * (x1 - x2)
                    temp_list.append(temp_num)
        rmse_temp = math.sqrt(np.mean(temp_list))
        rmse_list.append(rmse_temp)
    main_rmse = np.mean(rmse_list)

    return main_rmse, rmse_list

def Kriging_Tensor_RMSE(kriging_tensor, test_tensor, test_tensor_mask):
    pre_tensor = kriging_tensor * test_tensor_mask
    [z, x, y] = pre_tensor.shape

    rmse_list = []

    for t in range(z):
        rmse_temp = Grid_Rmse(pre_tensor[t], test_tensor[t], test_tensor_mask[t])
        rmse_list.append(rmse_temp)
    main_rmse = np.mean(rmse_list)

    return main_rmse, rmse_list