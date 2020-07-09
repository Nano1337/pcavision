from matplotlib import pyplot as plt
import h5py
from lesion_extraction_2d.lesion_extractor_2d import get_train_data
from skimage.feature import greycomatrix, greycoprops
if __name__ == "__main__":
    h5_file = h5py.File('C:\\Users\\haoli\\Documents\\pcavision\\hdf5_create\\prostatex-train-ALL.hdf5', 'r')

    X, y, attr = get_train_data(h5_file, ['ADC']) #gets all images of specified type
    print("finished sectioning")
    #X is NumPy array with all lesions so X[n] for 2d array for one lesion
    #y is clinical significance as True/False
    #attr is the dictionary of metadata associated with the region
    #change pixel size in 2 places in lesion_extractor_2d for different bounding box size
    # n = 5 #lesion number
    # min = 6 #minimum intensity
    #print(type(y[1])) #Clinical Significance as True/False
    print(attr[0])
    clinsigtrue, clinsigfalse = [], []
    num = 0
    for patch in X:
        if 'TRUE' in attr[num]['ClinSig'].decode('UTF-8'):
            clinsigtrue.append(patch)
            num += 1
        else:
            clinsigfalse.append(patch)
            num += 1
    # print(attr[n].get("Zone")) #dictionary of metadata
    # # X[n][X[n] > min] = 10
    # plt.imshow(X[n], cmap='gray')
    # plt.show()
    # print(X[n])
    xs = []
    ys = []
    # computer some GLCM properties for each prop
    # may throw error bc 1 patch has value of None
    i = 0
    for patch in (clinsigtrue + clinsigfalse):
        if patch is None:
            print("found None")
            i += 1
            # continue
        else:
            glcm = greycomatrix(patch, distances=[1], angles=[0], levels=3350, symmetric=True, normed=True)
            x = greycoprops(glcm, 'dissimilarity')[0, 0]
            y = greycoprops(glcm, 'correlation')[0, 0]
            xs.append(x)
            ys.append(y)
            print("processing " + str(i))
            print(x)
            print(y)
            i += 1
    # create the figure
    fig = plt.figure(figsize=(8, 8))
    print("begin plotting")
    # for each patch, plot (dissimilarity, correlation)
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(xs[:len(clinsigtrue)], ys[:len(clinsigtrue)], 'go',
            label='True')
    ax.plot(xs[len(clinsigtrue):], ys[len(clinsigtrue):], 'bo',
            label='False')
    ax.set_xlabel('GLCM Dissimilarity')
    ax.set_ylabel('GLCM Correlation')
    ax.legend()
    fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()