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


def get_train_data(h5_file, query_words, imagenum, size_px=16 ):
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
    h5_file = h5py.File('C:\\Users\\haoli\\Documents\\pcavision\\hdf5_create\\prostatex-train-ALL.hdf5', 'r')

    X, y, attr = get_train_data(h5_file, ['ADC'], -11) #gets all images of specified type
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
    n = 181
    plt.imshow(X[n][52:59, 30:50], cmap='gray') #[y, x]
    print(attr[n]) #dictionary of metadata
    # ax = plt.gca()
    # rect = patches.Rectangle((50, 60), 6, 13, linewidth=1, edgecolor='cyan', fill=False)
    # ax.add_patch(rect)
    plt.show()
    #print(np.shape(X[n]))

    # THIS WAS A PAIN TO MAKE
    # ADC TZ X[0][60:75, 35:50] image[-10]
    # ADC TZ X[2][60:68, 34:46] image[-10]
    # ADC TZ X[15][62:75, 33:50] image[-10]
    # ADC TZ X[18][55:75, 30:50] image[-10]
    # ADC TZ X[25][55:70, 30:45] image[-10]
    # ADC TZ X[37][55:70, 35:50] image[-10]
    # ADC TZ X[39][60:70, 30:45] image[-10]
    # ADC TZ X[41][57:70, 35:50] image[-10]
    # ADC TZ X[50][56:75, 31:50] image[-10]
    # ADC TZ X[55][60:70, 35:50] image[-10]
    # ADC TZ X[58][58:65, 38:48] image[-10]
    # ADC TZ X[68][52:65, 33:50] image[-10]
    # ADC TZ X[70][51:73, 30:55] image[-10]
    # ADC TZ X[80][56:80, 30:55] image[-10]
    # ADC TZ X[83][55:70, 30:50] image[-10]

    # ADC AS X[2][50:60, 32:45] image[-10]
    # ADC AS X[21][50:56, 35:48] image[-10]
    # ADC AS X[49][50:56, 37:46] image[-10]
    # ADC AS X[82][42:50, 25:50] image[-10]
    # ADC AS X[112][55:62, 32:45] image[-9]
    # ADC AS X[137][48:56, 40:52] image[-11]
    # ADC AS X[139][52:60, 34:48] image[-8]
    # ADC AS X[164][50:60, 30:50] image[-11]
    # ADC AS X[174][55:65, 28:50] image[-11]
    # ADC AS X[181][52:59, 30:50] image[-11]

    # T2WI TZ X[0][180:205, 175:210] image[-10]
    # T2WI TZ X[9][160:200, 150:230] image[-10]
    # T2WI TZ X[12][150:200, 145:200] image[-10]
    # T2WI TZ X[13][160:200, 145:200] image[-10]
    # T2WI TZ X[14][167:183, 170:210] image[-10]
    # T2WI TZ X[21][165:220, 150:220] image[-10]
    # T2WI TZ X[23][150:210, 150:250] image[-10]
    # T2WI TZ X[25][145:220, 150:230] image[-10]
    # T2WI TZ X[45][170:210, 150:220] image[-10]
    # T2WI TZ X[53][170:220, 165:230] image[-10]
    # T2WI TZ X[55][165:210, 170:230] image[-10]
    # T2WI TZ X[63][140:180, 160:220] image[-10]
    # T2WI TZ X[68][150:200, 160:230] image[-10]
    # T2WI TZ X[81][160:230, 150:220] image[-10]
    # T2WI TZ X[83][160:230, 150:240] image[-10]

    # T2WI AS X[0][150:180, 165:200] image[-10]
    # T2WI AS X[2][145:165, 170:200] image [-10]
    # T2WI AS X[3][150:170, 167:210] image[-10]
    # T2WI AS X[22][160:176, 160:200] image[-10]
    # T2WI AS X[29][150:166, 165:200] image[-10]
    # T2WI AS X[30][150:168, 170:210] image[-10]
    # T2WI AS X[49][135:151, 180:215] image[-10]
    # T2WI AS X[54][148:170, 170:220] image[-10]
    # T2WI AS X[73][140:160, 170:220] image[-10]
    # T2WI AS X[79][155:170, 165:210] image[-10]

    # ADC PZ X[0][58:72, 50:56] image[-11]
    # ADC PZ X[4][73:80, 28:38] image[-8]
    # ADC PZ X[7][70:78, 35:50] image[-7]
    # ADC PZ X[8][70:78, 35:50] image[-7]
    # ADC PZ X[11][70:76, 47:57] image[-11]
    # ADC PZ X[16][69:75, 45:55] image[-10]
    # ADC PZ X[17][69:76, 45:55] image[-10]
    # ADC PZ X[21][72:80, 30:50] image[-10]
    # ADC PZ X[36][70:80, 43:55] image[-10]
    # ADC PZ X[51][66:73, 27:35] image[-8]
    # ADC PZ X[57][60:75, 51:59] image[-9]
    # ADC PZ X[58][57:75, 50:59] image[-9]
    # ADC PZ X[59][57:75, 51:59] image[-9]
    # ADC PZ X[76][67:80, 33:53] image[-12]
    # ADC PZ X[84][63:74, 48:54] image[-10]
    # ADC PZ X[92][60:72, 27:35] image[-10]

    # T2WI PZ X[0][195:211, 220:245] image[-10]
    # T2WI PZ X[6][237:252, 220:248] image[-8]
    # T2WI PZ X[10][215:231, 160:180] image[-10]
    # T2WI PZ X[11][225:241, 160:180] image[-10]
    # T2WI PZ X[16][200:220, 220:240] image[-9]
    # T2WI PZ X[17][200:230, 220:240] image[-9]
    # T2WI PZ X[21][223:250, 193:230] image[-9]
    # T2WI PZ X[22][218:235, 140:165] image[-10]
    # T2WI PZ X[24][190:225, 150:166] image[-10]
    # T2WI PZ X[36][220:250, 210:240] image[-10]
    # T2WI PZ X[51][213:230, 145:180] image[-8]
    # T2WI PZ X[57][170:212, 225:250] image[-9]
    # T2WI PZ X[58][170:212, 225:250] image[-9]
    # T2WI PZ X[59][170:212, 225:250] image[-9]
    # T2WI PZ X[76][224:240, 160:230] image[-12]
    # T2WI PZ X[84][210:230, 150:180] image[-10]




