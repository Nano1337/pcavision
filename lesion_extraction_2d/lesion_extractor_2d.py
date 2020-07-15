import math
import numpy as np
import h5py
from lesion_extraction_2d.h5_query import get_lesion_info
from medpy.filter import IntensityRangeStandardization
import pickle
class Centroid:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)

def extract_lesion_2d(img, centroid_position, size=None, realsize=9, imagetype='ADC'):
    if imagetype == 'T2TRA':
        if size is None:
            sizecal = math.ceil(realsize * 2)
    elif imagetype == 'KTRANS':
        if size is None:
            sizecal = math.ceil(realsize / 1.5)
    elif imagetype == 'ADC':
        if size is None:
            sizecal = math.ceil(realsize / 2)
        else:
            sizecal = size
    else:
        sizecal = size
    x_start = int(centroid_position.x - sizecal / 2)
    x_end = int(centroid_position.x + sizecal / 2)
    y_start = int(centroid_position.y - sizecal / 2)
    y_end = int(centroid_position.y + sizecal / 2)

    if centroid_position.z < 0 or centroid_position.z >= len(img):
        return None

    img_slice = img[centroid_position.z]

    return img_slice[y_start:y_end, x_start:x_end]


def parse_centroid(ijk):
    coordinates = ijk.split(b" ")
    return Centroid(int(coordinates[0]), int(coordinates[1]), int(coordinates[2]))


def get_train_data(h5_file, query_words, size_px=9):
    lesion_info = get_lesion_info(h5_file, query_words)

    X = []
    y = []
    lesion_attributes = []
    previous_patient = ''
    for infos, image in lesion_info:
        current_patient = infos[0]['name'].split('/')[1]
        if current_patient == previous_patient:
            print('Warning in {}: Found duplicate match for {}. Skipping...'
                    .format(get_train_data.__name__, current_patient))
            continue
        for lesion in infos:

            centroid = parse_centroid(lesion['ijk'])

            if centroid.z < 0 or centroid.z >= len(image):
                lesion_img = None
            else:
                #lesion_img = image[centroid.z] #get full mri
                lesion_img = extract_lesion_2d(image, centroid, size=size_px) #to crop lesion from centroid
            if lesion_img is None:
                lesion_img = X[0]
                print('Warning in {}: ijk out of bounds for {}. No lesion extracted'
                        .format(get_train_data.__name__, lesion))
            X.append(lesion_img)
            lesion_attributes.append(lesion)

            y.append(lesion['ClinSig'] == b"TRUE")

        previous_patient = current_patient
    X = np.asarray(X)
    if 't2_tse_tra' in query_words:
        irs = IntensityRangeStandardization()
        _, X = irs.train_transform(X)
        X = [pixels.astype(int) + 1000 for pixels in X]
    return X, np.asarray(y), np.asarray(lesion_attributes)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    """ Example usage: """
    h5_file = h5py.File('C:\\Users\\haoli\\Documents\\pcavision\\hdf5_create\\prostatex-train-ALL.hdf5', 'r')

    X, y, attr = get_train_data(h5_file, ['ADC']) #gets all images of specified type
    # irs = IntensityRangeStandardization()
    # trained_model, transformed_images = irs.train_transform(X)
    # with open('my_trained_model.pkl', 'wb') as f:
    #     pickle.dump(irs, f)

    n = 0 #lesion number
    min = 6 #minimum intensity
    print(type(X[1]))
    print(y[n]) #Clinical Significance as True/False
    print(attr[n]) #dictionary of metadata
    # X[n][X[n] > min] = 10
    # ax = plt.hist(X[n].ravel(), bins = 256)
    plt.imshow(X[n], cmap='gray')
    plt.show()
    print(X[n])
