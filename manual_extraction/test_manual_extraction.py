from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from lesion_extraction_2d.lesion_extractor_2d import get_train_data
import h5py
from sklearn.feature_extraction.image import extract_patches_2d as eptwo

if __name__ == '__main__':

    # Extract numbers from file
    with open('test_manual_negatives.txt', 'r') as fin:
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
    with open('test_manual_negatives.txt', 'r') as fin:
        h5_file = h5py.File('C:\\Users\\haoli\\Documents\\pcavision\\hdf5_create\\prostatex-test-ALL.hdf5', 'r')

        content = fin.readlines()
        i = 0
        ln = 0
        adcpz = np.zeros((17, 20, 6, 6))
        t2pz = np.zeros((17, 20, 16, 16))
        adctz = np.zeros((12, 15, 6, 6))
        t2tz = np.zeros((12, 15, 16, 16))
        adcas = np.zeros((7, 14, 6, 6))
        t2as = np.zeros((7, 14, 16, 16))
        for line in content:
            print("currently on line " + str(ln))
            if "ADC PZ" in line:
                X, _, _ = get_train_data(h5_file, ['ADC'], numarr[i, 5])
                slice = X[numarr[i, 0]][numarr[i, 1]:numarr[i, 2], numarr[i, 3]: numarr[i, 4]]
                temp = eptwo(slice, (6, 6), max_patches=20)
                while np.shape(temp)[0] <= 20:
                    temp = np.vstack((temp, eptwo(slice, (6, 6), max_patches=20)))
                temp = temp[:20]
                np.random.shuffle(temp)
                adcpz[i] = temp
            if "T2WI PZ" in line:
                if "T2WI PZ X[10][200:230, 130:160] image[-10]" in line:
                    i = 0
                X, _, _ = get_train_data(h5_file, ['t2_tse_tra'], numarr[ln, 5])
                slice = X[numarr[ln, 0]][numarr[ln, 1]:numarr[ln, 2], numarr[ln, 3]: numarr[ln, 4]]
                temp = eptwo(slice, (16, 16), max_patches=20)
                while np.shape(temp)[0] <= 20:
                    temp = np.vstack((temp, eptwo(slice, (16, 16), max_patches=20)))
                temp = temp[:20]
                np.random.shuffle(temp)
                t2pz[i] = temp
            if "ADC TZ" in line:
                if "ADC TZ X[12][60:68, 60:68] image[-10]" in line:
                    i = 0
                X, _, _ = get_train_data(h5_file, ['ADC'], numarr[ln, 5])
                slice = X[numarr[ln, 0]][numarr[ln, 1]:numarr[ln, 2], numarr[ln, 3]: numarr[ln, 4]]
                temp = eptwo(slice, (6, 6), max_patches=15)
                while np.shape(temp)[0] <= 15:
                    temp = np.vstack((temp, eptwo(slice, (6, 6), max_patches=15)))
                temp = temp[:15]
                np.random.shuffle(temp)
                adctz[i] = temp
            if "T2WI TZ" in line:
                if "T2WI TZ X[12][175:200, 170:210] image[-9]" in line:
                    i = 0
                X, _, _ = get_train_data(h5_file, ['t2_tse_tra'], numarr[ln, 5])
                slice = X[numarr[ln, 0]][numarr[ln, 1]:numarr[ln, 2], numarr[ln, 3]: numarr[ln, 4]]
                temp = eptwo(slice, (16, 16), max_patches=15)
                while np.shape(temp)[0] <= 15:
                    temp = np.vstack((temp, eptwo(slice, (16, 16), max_patches=15)))
                temp = temp[:15]
                np.random.shuffle(temp)
                t2tz[i] = temp
            if "ADC AS" in line:
                if "ADC AS X[27][42:48, 60:70] image[-10]" in line:
                    i = 0
                X, _, _ = get_train_data(h5_file, ['ADC'], numarr[ln, 5])
                slice = X[numarr[ln, 0]][numarr[ln, 1]:numarr[ln, 2], numarr[ln, 3]: numarr[ln, 4]]
                temp = eptwo(slice, (6, 6), max_patches=14)
                while np.shape(temp)[0] <= 14:
                    temp = np.vstack((temp, eptwo(slice, (6, 6), max_patches=14)))
                temp = temp[:14]
                np.random.shuffle(temp)
                adcas[i] = temp
            if "T2WI AS" in line:
                if "T2WI AS X[12][140:170, 170:210] image[-10]" in line:
                    i = 0
                X, _, _ = get_train_data(h5_file, ['t2_tse_tra'], numarr[ln, 5])
                slice = X[numarr[ln, 0]][numarr[ln, 1]:numarr[ln, 2], numarr[ln, 3]: numarr[ln, 4]]
                temp = eptwo(slice, (16, 16), max_patches=14)
                while np.shape(temp)[0] <= 14:
                    temp = np.vstack((temp, eptwo(slice, (16, 16), max_patches=14)))
                temp = temp[:14]
                np.random.shuffle(temp)
                t2as[i] = temp
            i += 1
            ln += 1
        np.save('testadcpz.npy', adcpz)
        np.save('testt2pz.npy', t2pz)
        np.save('testadctz.npy', adctz)
        np.save('testt2tz.npy', t2tz)
        np.save('testadcas.npy', adcas)
        np.save('testt2as.npy', t2as)