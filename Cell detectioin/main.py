import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used

from train_cell_detection import run
from get_training_hyperparms import get_hyper_params

if __name__ == '__main__':
    hyper_params = get_hyper_params()
    for _, param in hyper_params.items():
        run(**param, all_params=param)
