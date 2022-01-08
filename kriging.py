################################################################
# Demonstration of Semivariogram Modeling and Ordinary Kriging
# Written by Koya SATO
# 2019.07.04 ver.1.0
# Verification:
# - Windows10 Home x64
# - Python     3.7.3
# - numpy      1.16.2
# - scipy      1.2.1
# - matplotlib 3.0.3
################################################################

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import math
import random
from metrics import Grid_Rmse
import time

''' sampling via multivariate normal distribution with no trend '''
def genMultivariate(data,grid_num):
    #print("--genMultivariate()--")
    z = []
    for i in range(len(grid_num)):
        for j in range(len(grid_num[i])):
            if grid_num[i][j] == 1:
                z.append(data[i][j])
    z = np.array(z)
    return z

'''measurement location '''
def genMeasurementLocation(grid_num):
    x = []
    y = []
    #print("--genMeasurementLocation()--")
    for i in range(len(grid_num)):
        for j in range(len(grid_num[i])):
            if grid_num[i][j] == 1:
                x.append(i)
                y.append(j)
    x = np.array(x)
    y = np.array(y)
    return x, y

''' gen empirical semivariogram via binning '''
def genSemivar(data, d_max, num,N):
    def genCombinations(arr):
      r, c = np.triu_indices(len(arr), 1)
      return np.stack((arr[r], arr[c]), 1)

    #print("--genSemivar()--")
    d_semivar = np.linspace(0.0, d_max, num)
    SPAN = d_semivar[1] - d_semivar[0]

    indx = genCombinations(np.arange(N))
    
    d = distance(data[indx[:, 0], 0], data[indx[:, 0], 1], data[indx[:, 1], 0], data[indx[:, 1], 1])
    indx = indx[d<=d_max]
    d = d[d <= d_max]
    semivar = (data[indx[:, 0], 2] - data[indx[:, 1], 2])**2

    semivar_avg = np.empty(num)
    for i in range(num):
        d1 = d_semivar[i] - 0.5*SPAN
        d2 = d_semivar[i] + 0.5*SPAN
        indx_tmp = (d1 < d) * (d <= d2) #index within calculation span
        semivar_tr = semivar[indx_tmp]
        if len(semivar_tr) != 0:
            semivar_avg[i] = semivar_tr.mean()
        else:
            semivar_avg[i] = np.nan

    return d_semivar[np.isnan(semivar_avg) == False], 0.5 * semivar_avg[np.isnan(semivar_avg) == False]

'''theoretical semivariogram (exponential)'''
def semivar_exp(d, nug, sill, ran):
    return np.abs(nug) + np.abs(sill) * (1.0-np.exp(-d/(np.abs(ran))))

'''fitting emperical semivariotram to theoretical model'''
def semivarFitting(d, data):
    def objFunc(x):
        theorem = semivar_exp(d, x[0], x[1], x[2])
        return ((data-theorem)**2).sum()

    x0 = np.random.uniform(0.0, 1.0, 3)
    res = minimize(objFunc, x0, method='nelder-mead')
    for i in range(5):
        x0 = np.random.uniform(0.0, 1.0, 3)
        res_tmp = minimize(objFunc, x0, method='nelder-mead')
        if res.fun > res_tmp.fun:
            res = res_tmp
    return np.abs(res.x)

def ordinaryKriging(mat, x_vec, y_vec, z_vec, x_rx, y_rx, nug, sill, ran):
    vec = np.ones(len(z_vec)+1, dtype=np.float)

    d_vec = distance(x_vec, y_vec, x_rx, y_rx)
    vec[:len(z_vec)] = semivar_exp(d_vec, nug, sill, ran)
    weight = np.linalg.solve(mat, vec)
    est = (z_vec * weight[:len(z_vec)]).sum()

    return est

def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

'''matrix for weight calculation in Ordinary Kriging'''
def genMat(x_vec, y_vec, z_vec, nug, sill, ran):
    mat = distance(x_vec, y_vec, x_vec[:, np.newaxis], y_vec[:, np.newaxis])
    mat = semivar_exp(mat, nug, sill, ran)
    mat = np.vstack((mat, np.ones(len(z_vec))))
    mat = np.hstack((mat, np.ones([len(z_vec)+1, 1])))
    mat[len(z_vec)][len(z_vec)] = 0.0

    return mat

def Find_Sample_Num(grid_num):
    k = 0
    for i in range(len(grid_num)):
        for j in range(len(grid_num[i])):
            if grid_num[i][j] == 1:
                k = k+1
    return k

def data_reconstruct(data,grid_num,max_num,min_num):
    R = [[0 for i in range(len(grid_num[0]))] for j in range(len(grid_num))]
    for i in range(len(grid_num)):
        for j in range(len(grid_num[i])):
            if(R[i][j] < 0):
                R[i][j] = min_num
            else:
                R[i][j] = data[i][j]*(max_num-min_num)+min_num
    return R

def Test_Grid(data,width,height,point_num):
    ret_grid = [[0 for i in range(height)] for j in range(width)]
    ret_data = [[0 for i in range(height)] for j in range(width)]
    x = []
    y = []
    for k in range(point_num):
        x.append(random.randint(0,width-1))
    for k in range(point_num):
        y.append(random.randint(0,height-1))
    for k in range(point_num):
        ret_grid[x[k]][y[k]] = 1
        ret_data[x[k]][y[k]] = data[x[k]][y[k]]
    return ret_data,ret_grid

def Intigrate_Z(z):
    ret_data = [[0 for i in range(len(z[0][0]))] for j in range(len(z[0]))]
    for i in range(len(ret_data)):
        for j in range(len(ret_data[i])):
            ret_data[i][j] = 0.5 * z[0][i][j] + 0.5 * (z[1][i][j] + z[2][i][j])
    return ret_data

def data_process(data,grid_num):
    R = [[0 for i in range(len(grid_num[0]))] for j in range(len(grid_num))]
    for i in range(len(grid_num)):
        for j in range(len(grid_num[i])):
            if grid_num[i][j] != 1:
                grid_num[i][j] = 0
    max_num,min_num = get_max_min_num(data,grid_num)
    for i in range(len(grid_num)):
        for j in range(len(grid_num[i])):
            if grid_num[i][j] == 1:
                if max_num-min_num == 0:
                    R[i][j] = 0
                else:
                    R[i][j] = (data[i][j]-min_num)/(max_num-min_num)
            else:
                R[i][j] = 0
    R = np.array(R)*np.array(grid_num)
    return R,max_num,min_num

def get_max_min_num(data,grid_num):
    max_num = -10000
    min_num = 10000
    for i in range(len(grid_num)):
        for j in range(len(grid_num[i])):
            if grid_num[i][j] == 1:
                if data[i][j] == -1000:
                    print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print(i,j)
                if data[i][j] > max_num:
                    max_num = data[i][j]
                if data[i][j] < min_num:
                    min_num = data[i][j]
    return  max_num,min_num

def kriging_insert(real_data, grid_num, max_distance = 10, max_point = 10):
    #real_matrix,max_num,min_num = data_process(real_data,grid_num)
    real_matrix = real_data.copy()

    '''measurement configuration'''
    WIDTH_AREA = len(grid_num) #area length [m]
    HEIGHT_AREA = len(grid_num[0])

    N = Find_Sample_Num(grid_num) #number of samples

    '''parameters for semivariogram modeling'''
    #D_MAX = int(WIDTH_AREA/2) #maximum distance in semivariogram modeling
    #N_SEMIVAR = int(WIDTH_AREA/2) #number of points for averaging empirical semivariograms

    D_MAX = max_distance #maximum distance in semivariogram modeling
    N_SEMIVAR = max_point #number of points for averaging empirical semivariograms

    '''get measurement dataset'''
    x, y = genMeasurementLocation(grid_num) #get N-coodinates for measurements
    z = genMultivariate(real_matrix,grid_num) #gen measurement samples based on multivariate normal distribution

    '''get empirical semivariogram model'''
    data = np.vstack([x, y, z]).T
    d_sv, sv = genSemivar(data, D_MAX, N_SEMIVAR,N)
    param = semivarFitting(d_sv, sv)

    '''Ordinary Kriging'''
    x_valid = np.linspace(0, WIDTH_AREA, WIDTH_AREA)
    y_valid = np.linspace(0, HEIGHT_AREA, HEIGHT_AREA)
    X, Y = np.meshgrid(x_valid, y_valid)
    z_map = np.zeros([len(x_valid), len(y_valid)])

    mat = genMat(x, y, z, param[0], param[1], param[2])
    for i in range(len(x_valid)):
        for j in range(len(y_valid)):
            z_map[i][j] = ordinaryKriging(mat, x, y, z, x_valid[i], y_valid[j], param[0], param[1], param[2])

    # z_temp = np.array(z_map).T
    # '''plot results'''
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1, adjustable='box', aspect=1.0)
    # ax.pcolor(X, Y ,z_temp, cmap='jet')
    # ax.set_title('1')
    # plt.show()

    #z_map = data_reconstruct(z_map,grid_num,max_num,min_num)
    return np.array(z_map)

def Kriging_Tensor_Completion(day, train_tensor, train_grid, max_distance = 10):
    print()
    print('************************ Day: {0} Kriging **************************'.format(day))
    start = time.time()
    completion_tensor = train_tensor.copy()
    [z, x, y] = train_tensor.shape

    for t in range(z):
        temp_arr = kriging_insert(train_tensor[t], train_grid[t], max_distance = max_distance)
        completion_tensor[t, :, :] = temp_arr
    end = time.time()
    print('Kriging running time: %d seconds'%(end - start))
    print('*********************************************************')
    print()
    return completion_tensor, end-start

