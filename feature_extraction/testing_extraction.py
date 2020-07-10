from matplotlib import pyplot as plt
import h5py
import numpy as np
from numpy import asarray, save
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from lesion_extraction_2d.lesion_extractor_2d import get_train_data
from skimage.feature import greycomatrix, greycoprops
import functools
import pickle

@functools.lru_cache(maxsize=None)
def get_feature_extracts():
    """
    Returns lesion n x features d matrix, binary label vector, and {column # : feature name} dictionary
    """
    # Note: change pixel size in 2 places in lesion_extractor_2d for different bounding box size (currently 9x9)
    # load hdf5 file
    h5_file = h5py.File('C:\\Users\\haoli\\Documents\\pcavision\\hdf5_create\\prostatex-train-ALL.hdf5', 'r')

    # extract info for matching MRI type name
    X, y, attr = get_train_data(h5_file, ['ADC'])  # gets all images of specified type
    # X is NumPy array with all lesions so X[n] is the 2d array for 1 lesion
    # y is clinical significance as True/False
    # attr is the dictionary of metadata associated with the region
    print("lesion extraction complete")
    print("starting feature extraction")

    # Initiate Clinical Significance truth vector
    clinsig_vector = get_clinsig_vector(X, attr)

    # with zone info
    feature_matrix = get_zone(X)
    feature_dict = { 0 : "zone",
                     1 : "zone",
                     2 : "zone",
                     3 : "ADC GLCM dissimilarity",
                     4 : "ADC GLCM correlation",
                     5 : "ADC GLCM contrast",
                     6 : "ADC GLCM homogeneity",
                     7 : "ADC GLCM energy",
                     8 : "ADC GLCM angular second moment",
                     9 : "ADC 10%",
                     10: "ADC average"}

    # get feature vectors
    dissimilarity, correlation, contrast, homo, energy, asm = get_GLCM(X)
    adc10vect, adcmeanvect = get_ADC_stats(X)

    # concatenate feature vectors column-wise
    feature_matrix = np.concatenate((feature_matrix, dissimilarity), axis=1)
    feature_matrix = np.concatenate((feature_matrix, correlation), axis=1)
    feature_matrix = np.concatenate((feature_matrix, contrast), axis=1)
    feature_matrix = np.concatenate((feature_matrix, homo), axis=1)
    feature_matrix = np.concatenate((feature_matrix, energy), axis=1)
    feature_matrix = np.concatenate((feature_matrix, asm), axis=1)
    feature_matrix = np.concatenate((feature_matrix, adc10vect), axis=1)
    feature_matrix = np.concatenate((feature_matrix, adcmeanvect), axis=1)
    return feature_matrix, clinsig_vector, feature_dict

def get_zone(X):
    # create intitial dataframe
    zone_types = ('AS', 'SV', 'PZ', 'TZ')
    zone_df = pd.DataFrame(zone_types, columns=['Zone_Types'])

    # creating instance of labelencoder
    labelencoder = LabelEncoder()

    # assigning numerical values and storing in another column
    zone_df['Zone_Types_Cat'] = labelencoder.fit_transform(zone_df['Zone_Types'])

    # create instance of one-hot-encoder
    enc = OneHotEncoder(handle_unknown='ignore')

    # passing zone-types-cat column (label encoded values of zone_types)
    enc_df = pd.DataFrame(enc.fit_transform(zone_df[['Zone_Types_Cat']]).toarray()[:, 1:])

    # merge with main df zone_df on key values
    zone_df = zone_df.join(enc_df)
    print(zone_df)
    num = 0
    zvect = np.zeros((len(X), 3))
    for patch in X:
        zone = attr[num]['Zone'].decode('UTF-8')
        row = zone_df.loc[zone_df['Zone_Types'] == zone]
        # 1-hot encode zone categorical data
        """
                  Zone_Types  Zone_Types_Cat    0    1    2
        0         AS               0            0.0  0.0  0.0
        1         SV               2            0.0  1.0  0.0
        2         PZ               1            1.0  0.0  0.0
        3         TZ               3            0.0  0.0  1.0
        """
        zvect[num, 0] = int(row.iloc[:, -3].tolist()[-1])
        zvect[num, 1] = int(row.iloc[:, -2].tolist()[-1])
        zvect[num, 2] = int(row.iloc[:, -1].tolist()[-1])
        num += 1
    return zvect

def get_GLCM(X):
    # initiate GLCM dissimilarity and correlation vectors
    dissimilarity = np.zeros((len(X), 1))
    correlation = np.zeros((len(X), 1))
    contrast = np.zeros((len(X), 1))
    homo = np.zeros((len(X), 1))
    energy = np.zeros((len(X), 1))
    asm = np.zeros((len(X), 1))
    # populate feature vectors with lesion patch information
    i = 0
    for patch in X:
        if patch is None:
            i += 1
            continue
        else:
            glcm = greycomatrix(patch, distances=[1], angles=[0], levels=3350, symmetric=True, normed=True)
            dissimilarity[i, 0] = greycoprops(glcm, 'dissimilarity')[0, 0]
            correlation[i, 0] = greycoprops(glcm, 'correlation')[0, 0]
            contrast[i, 0] = greycoprops(glcm, 'contrast')[0, 0]
            homo[i, 0] = greycoprops(glcm, 'homogeneity')[0, 0]
            energy[i, 0] = greycoprops(glcm, 'energy')[0, 0]
            asm[i, 0] = greycoprops(glcm, 'ASM')[0, 0]
            print("Processing GLCM for lesion #" + str(i))
            i += 1

    # return vectors
    return dissimilarity, correlation, contrast, homo, energy, asm

def get_clinsig_vector(X, attr):
    num = 0
    csvector = np.zeros((len(X), 1))
    for patch in X:
        # 1-hot encode binary clinical significance, true = 1, false = 0
        if 'TRUE' in attr[num]['ClinSig'].decode('UTF-8'):
            csvector[num, 0] = 1
            num += 1
        else:
            csvector[num, 0] = 0
            num += 1
    return csvector

def get_ADC_stats(X):
    i = 0
    adc10vect = np.zeros((len(X), 1))
    adcmeanvect = np.zeros((len(X), 1))
    for patch in X:
        adc10vect[i, 0] = np.percentile(patch, 10)
        adcmeanvect[i, 0] = np.mean(patch)
        i += 1
    return adc10vect, adcmeanvect

if __name__ == "__main__":
    h5_file = h5py.File('C:\\Users\\haoli\\Documents\\pcavision\\hdf5_create\\prostatex-train-ALL.hdf5', 'r')

    # extract info for matching MRI type name
    X, y, attr = get_train_data(h5_file, ['ADC'])  # gets all images of specified type
    # csvector, num_positive = get_clinsig_vector(X, attr)
    # print(num_positive/len(X))
    a, b, c = get_feature_extracts() #takes 13 minutes to run
    print("Feature Matrix")
    print(a)
    print("Clinical Significance Vector")
    print(b)
    print("Feature Dictionary")
    print(c)

    # write numpy array and dictionary to files so only have to run program once
    save('feature_mat.npy', a)
    save('clinsig_vect.npy', b)
    with open('feature_dict.txt', 'wb') as handle:
        pickle.dump(c, handle)





    # Nothin to see here folks...
    # clinsigtrue, clinsigfalse = [], []
    # num = 0
    # for patch in X:
    #     if 'TRUE' in attr[num]['ClinSig'].decode('UTF-8'):  # gets clinical significance
    #         clinsigtrue.append(patch)
    #         num += 1
    #     else:
    #         clinsigfalse.append(patch)
    #         num += 1
    # xs = []
    # ys = []
    # # computer some GLCM properties for each prop
    # # may throw error bc 1 patch has value of None
    # i = 0
    # for patch in (clinsigtrue + clinsigfalse):
    #     if patch is None:
    #         print("found None")
    #         i += 1
    #         # continue
    #     else:
    #         glcm = greycomatrix(patch, distances=[1], angles=[0], levels=3350, symmetric=True, normed=True)
    #         x = greycoprops(glcm, 'dissimilarity')[0, 0]
    #         y = greycoprops(glcm, 'correlation')[0, 0]
    #         xs.append(x)
    #         ys.append(y)
    #         print("processing " + str(i))
    #         print(x)
    #         print(y)
    #         i += 1
    # # create the figure
    # fig = plt.figure(figsize=(8, 8))
    # print("begin plotting")
    # # for each patch, plot (dissimilarity, correlation)
    # ax = fig.add_subplot(3, 2, 2)
    # ax.plot(xs[:len(clinsigtrue)], ys[:len(clinsigtrue)], 'go',
    #         label='True')
    # ax.plot(xs[len(clinsigtrue):], ys[len(clinsigtrue):], 'bo',
    #         label='False')
    # ax.set_xlabel('GLCM Dissimilarity')
    # ax.set_ylabel('GLCM Correlation')
    # ax.legend()
    # fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
    # plt.tight_layout()
    # plt.show()
    # print(attr[n].get("Zone")) #dictionary of metadata and get zone
    # # X[n][X[n] > min] = 10
    # plt.imshow(X[n], cmap='gray')
    # plt.show()
    # print(X[n])
