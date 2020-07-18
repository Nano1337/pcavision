import h5py
from lesion_extraction_2d.lesion_extractor_2d import get_train_data
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d as eptwo

if __name__ == "__main__":
    h5_file = h5py.File('C:\\Users\\haoli\\Documents\\pcavision\\hdf5_create\\prostatex-train-ALL.hdf5', 'r')
    posadcpz = np.zeros((35, 9, 9))
    post2pz = np.zeros((35, 32, 32))
    posadctz = np.zeros((9, 9, 9))
    post2tz = np.zeros((9, 32, 32))
    posadcas = np.zeros((31, 9, 9))
    post2as = np.zeros((31, 32, 32))
    i, j, k = 0, 0, 0
    X, y, attr = get_train_data(h5_file, ['ADC'], 9)
    for pixarray, clinsig in zip(X, attr):
        csbool = clinsig['ClinSig'].decode('UTF-8')
        zone = clinsig['Zone'].decode('UTF-8')
        if csbool == 'TRUE':
            if zone == 'PZ':
                posadcpz[i] = pixarray
                i += 1
            if zone == 'TZ':
                posadctz[j] = pixarray
                j += 1
            if zone == 'AS':
                posadcas[k] = pixarray
                k += 1
    X, y, attr = get_train_data(h5_file, ['t2_tse_tra'], 32)
    i, j, k = 0, 0, 0
    for pixarray, clinsig in zip(X, attr):
        csbool = clinsig['ClinSig'].decode('UTF-8')
        zone = clinsig['Zone'].decode('UTF-8')
        if csbool == 'TRUE':
            if zone == 'PZ':
                post2pz[i] = pixarray
                i += 1
            if zone == 'TZ':
                post2tz[j] = pixarray
                j += 1
            if zone == 'AS':
                post2as[k] = pixarray
                k += 1
    print("finished initial compilation")
    adcpz = eptwo(posadcpz[0], (6, 6), max_patches=5)
    t2pz = eptwo(post2pz[0], (16, 16), max_patches=5)
    i = 1
    while i < 35:
        print("currently on: " + str(i))
        adcpz = np.vstack((adcpz, eptwo(posadcpz[i], (6, 6), max_patches=5)))
        t2pz = np.vstack((t2pz, eptwo(post2pz[i], (16, 16), max_patches=5)))
        i += 1
    np.save("posadcpz.npy", adcpz)
    np.save("post2pz.npy", t2pz)