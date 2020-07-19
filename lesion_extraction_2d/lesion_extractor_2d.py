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

def extract_lesion_2d(img, centroid_position, size=None, realsize=16, imagetype='ADC'):
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


def get_train_data(h5_file, query_words, imagenum, size_px=16):
    # file = 'C:\\Users\\haoli\\Documents\\pcavision\\lesion_extraction_2d\\t2_tse_tra_train.txt'
    # with open(file, 'rb') as fp:
    #     lesion_info = pickle.load(fp)
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
                lesion_img = image[imagenum] #get full mri
                #lesion_img = extract_lesion_2d(image, centroid, size=size_px) #to crop lesion from centroid
            if lesion_img is None:
                lesion_img = X[0]
                print('Warning in {}: ijk out of bounds for {}. No lesion extracted'
                        .format(get_train_data.__name__, lesion))
            X.append(lesion_img)
            lesion_attributes.append(lesion)

            y.append(lesion['ClinSig'] == b"TRUE")

        previous_patient = current_patient
    X = np.asarray(X)
    # if 't2_tse_tra' in query_words:
    #     irs = IntensityRangeStandardization()
    #     _, X = irs.train_transform(X)
    #     X = [pixels.astype(int) + 1000 for pixels in X]
    return X, np.asarray(y), np.asarray(lesion_attributes)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import matplotlib.patches as patches
    """ Example usage: """
    h5_file = h5py.File('C:\\Users\\haoli\\Documents\\pcavision\\hdf5_create\\prostatex-test-ALL.hdf5', 'r')

    X, y, attr = get_train_data(h5_file, ['ADC'], -10) #gets all images of specified type
    n = 85
    plt.imshow(X[n][52:59, 42:50], cmap='gray')  # [y, x]
    print(attr[n])  # dictionary of metadata
    plt.show()

    # ADC PZ X[10][60:77, 43:60] image[-10]
    # ADC PZ X[27][60:68, 50:60] image[-10]
    # ADC PZ X[30][72:78, 45:55] image[-10]
    # ADC PZ X[32][47:60, 70:78] image[-10]
    # ADC PZ X[38][55:61, 53:62] image[-10]
    # ADC PZ X[45][52:60, 52:70] image[-10]
    # ADC PZ X[57][62:73, 51:57] image[-10]
    # ADC PZ X[65][65:80, 50:60] image[-10]
    # ADC PZ X[74][69:80, 54:60] image[-10]
    # ADC PZ X[101][67:75, 52:60] image[-10]
    # ADC PZ X[128][70:78, 44:53] image[-10]
    # ADC PZ X[128][70:78, 35:44] image[-10]
    # ADC PZ X[131][73:80, 25:33] image[-10]
    # ADC PZ X[137][68:74, 40:50] image[-10]
    # ADC PZ X[138][61:69, 50:58] image[-10]
    # ADC PZ X[140][68:75, 33:50] image[-10]
    # ADC PZ X[145][65:71, 24:35] image[-10]

    # ADC TZ X[12][60:68, 60:68] image[-10]
    # ADC TZ X[19][62:70, 50:65] image[-10]
    # ADC TZ X[22][62:70, 55:70] image[-10]
    # ADC TZ X[28][45:60, 40:80] image[-10]
    # ADC TZ X[29][60:66, 30:43] image[-10]
    # ADC TZ X[32][44:51, 58:72] image[-10]
    # ADC TZ X[55][47:60, 40:50] image[-10]
    # ADC TZ X[59][65:73, 35:48] image[-10]
    # ADC TZ X[60][55:65, 40:50] image[-10]
    # ADC TZ X[83][55:65, 32:45] image[-10]
    # ADC TZ X[102][58:65, 35:45] image[-10]
    # ADC TZ X[103][60:68, 40:55] image[-10]

    # ADC AS X[27][42:48, 60:70] image[-10]
    # ADC AS X[29][57:63, 30:38] image[-10]
    # ADC AS X[30][56:62, 35:43] image[-10]
    # ADC AS X[57][56:63, 39:49] image[-10]
    # ADC AS X[62][53:60, 40:50] image[-10]
    # ADC AS X[72][48:54, 36:44] image[-10]
    # ADC AS X[85][52:59, 42:50] image[-10]

    # T2WI PZ X[10][200:230, 130:160] image[-10]
    # T2WI PZ X[27][184:200, 125:150] image[-10]
    # T2WI PZ X[30][220:240, 210:233] image[-10]
    # T2WI PZ X[32][140:170, 185:202] image[-10]
    # T2WI PZ X[38][160:176, 125:150] image[-10]
    # T2WI PZ X[45][158:180, 192:208] image[-10]
    # T2WI PZ X[57][155:180, 190:205] image[-10]
    # T2WI PZ X[65][140:200, 200:218] image[-10]
    # T2WI PZ X[74][175:200, 195:210] image[-10]
    # T2WI PZ X[101][170:200, 187:203] image[-10]
    # T2WI PZ X[102][200:220, 150:170] image[-10]
    # T2WI PZ X[128][227:243, 195:230] image[-10]
    # T2WI PZ X[128][227:243, 160:195] image[-10]
    # T2WI PZ X[131][234:250, 130:165] image[-10]
    # T2WI PZ X[137][220:236, 215:235] image[-10]
    # T2WI PZ X[138][185:205, 230:250] image[-10]
    # T2WI PZ X[140][225:241, 150:180] image[-10]

    # T2WI TZ X[12][175:200, 170:210] image[-9]
    # T2WI TZ X[13][170:200, 170:210] image[-9]
    # T2WI TZ X[17][180:240, 160:200] image[-7]
    # T2WI TZ X[25][110:150, 120:180] image[-10]
    # T2WI TZ X[27][150:180, 130:180] image[-10]
    # T2WI TZ X[29][180:210, 150:210] image[-10]
    # T2WI TZ X[32][145:165, 140:180] image[-10]
    # T2WI TZ X[55][100:150, 158:177] image[-10]
    # T2WI TZ X[59][200:240, 170:220] image[-10]
    # T2WI TZ X[60][140:160, 140:180] image[-10]
    # T2WI TZ X[83][160:185, 172:200] image[-10]
    # T2WI TZ X[102][170:200, 160:210] image[-10]

    # T2WI AS X[12][140:170, 170:210] image[-10]
    # T2WI AS X[12][155:170, 170:210] image[-9]
    # T2WI AS X[29][160:176, 160:200] image[-10]
    # T2WI AS X[30][155:170, 175:200] image[-10]
    # T2WI AS X[57][134:150, 150:174] image[-10]
    # T2WI AS X[62][114:130, 145:175] image[-10]
    # T2WI AS X[72][96:112, 140:170] image[-10]

    # PZ, TZ, AS = 0, 0, 0
    # for zone in attr:
    #     if zone['Zone'].decode('UTF-8') == 'PZ':
    #         PZ += 1
    #     if zone['Zone'].decode('UTF-8') == 'TZ':
    #         TZ += 1
    #     if zone['Zone'].decode('UTF-8') == 'AS':
    #         AS += 1
    #
    # print(PZ, TZ, AS)

    # #compile zone pixel arrays from test
    # i = 0
    # test_pz_adc = np.zeros((113, 6, 6))
    # patchnum = 0
    # for patch in X:
    #     if attr[i]['Zone'].decode('UTF-8') == 'PZ':
    #         test_pz_adc[patchnum] = patch
    #         patchnum += 1
    #     i += 1
    #
    # np.save('test_pz_adc.npy', test_pz_adc)
    # print(test_pz_adc)

    # irs = IntensityRangeStandardization()
    # trained_model, transformed_images = irs.train_transform(X)
    # with open('my_trained_model.pkl', 'wb') as f:
    #     pickle.dump(irs, f)

    #n = 7#lesion number
    # min = 6 #minimum intensity
    # print(type(X[n]))
    # print(y) #Clinical Significance as True/False

    # X[n][X[n] > min] = 10
    # ax = plt.hist(X[n].ravel(), bins = 256)



    # ax = plt.gca()
    # rect = patches.Rectangle((50, 60), 6, 13, linewidth=1, edgecolor='cyan', fill=False)
    # ax.add_patch(rect)

    #print(np.shape(X[n]))






