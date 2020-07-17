from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from lesion_extraction_2d.lesion_extractor_2d import get_train_data
import h5py
from sklearn.feature_extraction.image import extract_patches_2d as eptwo

if __name__ == '__main__':

    # Extract numbers from file
    with open('manual_negatives.txt', 'r') as fin:
        content = fin.readlines()
        sdigit = ""
        line_entries = []
        for line in content:
            line_entry = []
            i = 0
            while i < len(line): # Didn't use for loop with range(len(line)) be range resets index even after updating
                while line[i:i+1].isdigit():
                    sdigit += line[i:i+1]
                    i += 1
                if len(sdigit) != 0:
                    line_entry.append(int(sdigit))
                    sdigit = ""
                i += 1
            if "T2WI" in line:
                line_entry = line_entry[1:]

            # pixel array slices are reversed for some reason
            line_entry[-1] = line_entry[-1]*-1

            line_entries.append(line_entry)
        numarr = np.array(line_entries)
    with open('manual_negatives.txt', 'r') as fin:
        h5_file = h5py.File('C:\\Users\\haoli\\Documents\\pcavision\\hdf5_create\\prostatex-train-ALL.hdf5', 'r')

        content = fin.readlines()
        i = 0
        ln = 0
        adcpz = np.zeros((16, 35, 6, 6))
        t2pz = np.zeros((16, 35, 16, 16))
        adctz = np.zeros((15, 16, 6, 6))
        t2tz = np.zeros((15, 16, 16, 16))
        adcas = np.zeros((10, 16, 6, 6))
        t2as = np.zeros((10, 16, 16, 16))
        for line in content:
            print("currently on line " + str(ln))
            if "ADC PZ" in line:
                X, _, _ = get_train_data(h5_file, ['ADC'], numarr[i, 5])
                slice = X[numarr[i, 0]][numarr[i, 1]:numarr[i, 2], numarr[i, 3]: numarr[i, 4]]
                temp = eptwo(slice, (6, 6), max_patches=35)
                while np.shape(temp)[0] <= 35:
                    temp = np.vstack((temp, eptwo(slice, (6, 6), max_patches=35)))
                temp = temp[:35]
                np.random.shuffle(temp)
                adcpz[i] = temp
            if "T2WI PZ" in line:
                if "T2WI PZ X[0][195:211, 220:245] image[-10]" in line:
                    i = 0
                X, _, _ = get_train_data(h5_file, ['t2_tse_tra'], numarr[ln, 5])
                slice = X[numarr[ln, 0]][numarr[ln, 1]:numarr[ln, 2], numarr[ln, 3]: numarr[ln, 4]]
                temp = eptwo(slice, (16, 16), max_patches=35)
                while np.shape(temp)[0] <= 35:
                    temp = np.vstack((temp, eptwo(slice, (16, 16), max_patches=35)))
                temp = temp[:35]
                np.random.shuffle(temp)
                t2pz[i] = temp
            if "ADC TZ" in line:
                if "ADC TZ X[0][60:75, 35:50] image[-10]" in line:
                    i = 0
                X, _, _ = get_train_data(h5_file, ['ADC'], numarr[ln, 5])
                slice = X[numarr[ln, 0]][numarr[ln, 1]:numarr[ln, 2], numarr[ln, 3]: numarr[ln, 4]]
                temp = eptwo(slice, (6, 6), max_patches=16)
                while np.shape(temp)[0] <= 16:
                    temp = np.vstack((temp, eptwo(slice, (6, 6), max_patches=16)))
                temp = temp[:16]
                np.random.shuffle(temp)
                adctz[i] = temp
            if "T2WI TZ" in line:
                if "T2WI TZ X[0][180:205, 175:210] image[-10]" in line:
                    i = 0
                X, _, _ = get_train_data(h5_file, ['t2_tse_tra'], numarr[ln, 5])
                slice = X[numarr[ln, 0]][numarr[ln, 1]:numarr[ln, 2], numarr[ln, 3]: numarr[ln, 4]]
                temp = eptwo(slice, (16, 16), max_patches=16)
                while np.shape(temp)[0] <= 16:
                    temp = np.vstack((temp, eptwo(slice, (16, 16), max_patches=16)))
                temp = temp[:16]
                np.random.shuffle(temp)
                t2tz[i] = temp
            if "ADC AS" in line:
                if "ADC AS X[2][50:60, 32:45] image[-10]" in line:
                    i = 0
                X, _, _ = get_train_data(h5_file, ['ADC'], numarr[ln, 5])
                slice = X[numarr[ln, 0]][numarr[ln, 1]:numarr[ln, 2], numarr[ln, 3]: numarr[ln, 4]]
                temp = eptwo(slice, (6, 6), max_patches=16)
                while np.shape(temp)[0] <= 16:
                    temp = np.vstack((temp, eptwo(slice, (6, 6), max_patches=16)))
                temp = temp[:16]
                np.random.shuffle(temp)
                adcas[i] = temp
            if "T2WI AS" in line:
                if "T2WI AS X[0][150:180, 165:200] image[-10]" in line:
                    i = 0
                X, _, _ = get_train_data(h5_file, ['t2_tse_tra'], numarr[ln, 5])
                slice = X[numarr[ln, 0]][numarr[ln, 1]:numarr[ln, 2], numarr[ln, 3]: numarr[ln, 4]]
                temp = eptwo(slice, (16, 16), max_patches=16)
                while np.shape(temp)[0] <= 16:
                    temp = np.vstack((temp, eptwo(slice, (16, 16), max_patches=16)))
                temp = temp[:16]
                np.random.shuffle(temp)
                t2as[i] = temp
            i += 1
            ln += 1
        np.save('adcpz.npy', adcpz)
        np.save('t2pz.npy', t2pz)
        np.save('adctz.npy', adctz)
        np.save('t2tz.npy', t2tz)
        np.save('adcas.npy', adcas)
        np.save('t2as.npy', t2as)