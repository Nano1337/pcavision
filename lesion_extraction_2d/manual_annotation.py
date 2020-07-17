from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from lesion_extraction_2d.lesion_extractor_2d import get_train_data
import h5py


def get_ADC_patch():
    patches = []
    return patches
if __name__ == '__main__':
    h5_file = h5py.File('C:\\Users\\haoli\\Documents\\pcavision\\hdf5_create\\prostatex-train-ALL.hdf5', 'r')
    # X, y, attr = get_train_data(h5_file, ['ADC'])
    # data = np.array(X[0])
    # img = plt.imshow(data)

