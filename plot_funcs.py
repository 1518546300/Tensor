from matplotlib import pyplot as plt
import numpy as np
import imageio, os

def grid_map_plot(day, index, map, name = 'pic'):
    WIDTH_AREA = len(map)
    HEIGHT_AREA = len(map[0])
    x_valid = np.linspace(0, WIDTH_AREA, WIDTH_AREA)
    y_valid = np.linspace(0, HEIGHT_AREA, HEIGHT_AREA)
    X, Y = np.meshgrid(x_valid, y_valid)
    map = map.T

    '''plot results'''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, adjustable='box', aspect=1.0)
    ax.pcolor(X, Y ,map, cmap='jet')
    ax.set_title(name)
    plt.show()

def grid_map_plot_save(day, index, map, file_name, name = 'pic'):
    WIDTH_AREA = len(map)
    HEIGHT_AREA = len(map[0])
    x_valid = np.linspace(0, WIDTH_AREA, WIDTH_AREA)
    y_valid = np.linspace(0, HEIGHT_AREA, HEIGHT_AREA)
    X, Y = np.meshgrid(x_valid, y_valid)
    map = map.T

    '''plot results'''
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, adjustable='box', aspect=1.0)
    ax.pcolor(X, Y ,map, cmap='jet')
    ax.set_title(name)
    #plt.show()
    if index < 10:
        pic_day_index_str = str(day) + "_0" + str(index)
    else:
        pic_day_index_str = str(day) + "_" + str(index)
    plt.savefig("F:\\大气环境\\任务_数据融合\\代码\\tensor\\图片\\" + file_name + "\\" + pic_day_index_str + ".png")

def tensor_map_plot(day, tensor, name = 'pic'):
    w, h, t = tensor.shape

    pic_row_num = 4
    pic_col_num = int(t/pic_row_num)

    WIDTH_AREA = w
    HEIGHT_AREA = h
    x_valid = np.linspace(0, WIDTH_AREA, WIDTH_AREA)
    y_valid = np.linspace(0, HEIGHT_AREA, HEIGHT_AREA)
    X, Y = np.meshgrid(x_valid, y_valid)
    fig = plt.figure()
    plt.title('---2019-day: ' + str(day + 1) + '---')
    for i in range(pic_row_num):
        for j in range(pic_col_num):
            map = tensor[:, :, i * pic_row_num + j].T
            ax_name = 'Time: ' + str(i * pic_col_num + j + 1)
            ax = fig.add_subplot(pic_row_num, pic_col_num, i * pic_col_num + j + 1, adjustable='box', aspect=1.0)
            ax.pcolor(X, Y ,map, cmap='jet')
            ax.set_title(ax_name)
    plt.show()

def png_2_gif(file_name, file_list, frequence_arr = [1, 2]):
    '''
        file_name under the main file path --- F:\\大气环境\\任务_数据融合\\代码\\tensor\\图片\\
    '''
    file_path = 'F:\\大气环境\\任务_数据融合\\代码\\tensor\\图片\\' + file_name + '\\'
    file_day_arr = file_list
    file_save_path = 'F:\\大气环境\\任务_数据融合\\代码\\tensor\\图片\\' + file_name + '\\gif\\'
    for i in range(len(file_day_arr)):
        images = []
        file_path_main = file_path + str(file_day_arr[i]) + "\\"
        filenames = sorted((fn for fn in os.listdir(file_path_main) if fn.endswith('.png')))
        for filename in filenames:
            images.append(imageio.imread(file_path_main + filename))
        for j in range(len(frequence_arr)):
            temp_images = images[0:len(images):frequence_arr[j]]
            imageio.mimsave(file_save_path + str(file_day_arr[i]) + '_' + str(frequence_arr[j]) + '.gif', temp_images, duration = frequence_arr[j], loop=1)

#png_2_gif('krige_plot', [251], frequence_arr = [1])