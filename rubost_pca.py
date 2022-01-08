from random import random
import tensorly as tl
from tensorly import decomposition
from funcs_kit import load_map_grid, split_data_set_2, day_tensor_data_gain, TensorMaxMin, data_gain

def r_pca():
    pass

if __name__ == '__main__':
    train_tensor_np, train_mask_tensor_np, train_grid, test_tensor_np, test_mask_tensor_np, test_grid = data_gain()
    train_tensor_obj = TensorMaxMin(train_tensor_np, train_grid)
    train_tensor = tl.tensor(train_tensor_obj.tensor)
    train_mask_tensor = tl.tensor(train_mask_tensor_np)
    D, E = decomposition.robust_pca(train_tensor, mask=train_mask_tensor, tol=1e-06, reg_E=1.0, reg_J=1.0, mu_init=0.0001, mu_max=10000000000.0, learning_rate=1.1, n_iter_max=100, verbose=1)

    D_reconstract = train_tensor_obj.Max_Min_Reconstruct(D)
    E_reconstract = train_tensor_obj.Max_Min_Reconstruct(E)

    recons_tensor = D_reconstract + E_reconstract

    r_pca()