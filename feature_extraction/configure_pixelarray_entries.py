import numpy as np
from sklearn.utils import shuffle
def linearize_pz():
    # negative samples
    unconfig_adcpz = np.load('C:\\Users\\haoli\\Documents\\pcavision\\manual_extraction\\testadcas.npy')
    unconfig_t2pz = np.load('C:\\Users\\haoli\\Documents\\pcavision\\manual_extraction\\testt2as.npy')

    #linearize ADC
    config_adcpz = np.zeros((98, 6, 6))
    i = 0
    for r in range(len(unconfig_adcpz)):
        for c in range(len(unconfig_adcpz[r])):
            config_adcpz[i] = unconfig_adcpz[r, c]
            i += 1

    #linearize T2WI
    config_t2pz = np.zeros((98, 16, 16))
    i = 0
    for r in range(len(unconfig_t2pz)):
        for c in range(len(unconfig_t2pz[r])):
            config_t2pz[i] = unconfig_t2pz[r, c]
            i += 1
    return config_adcpz, config_t2pz


if __name__ == "__main__":
    # unrandomized positive on top and negative on bottom
    post2pz = np.load('C:\\Users\\haoli\\Documents\\pcavision\\lesion_extraction_2d\\test_as_t2.npy')
    posadcpz = np.load('C:\\Users\\haoli\\Documents\\pcavision\\lesion_extraction_2d\\test_as_adc.npy')

    # print(np.shape(post2pz))
    # make unrandomized clinsig vect
    clinsig_vect = np.vstack((np.ones((34, 1)), np.zeros((98, 1))))

    # linearize negative pixel arrays
    negadcpz, negt2pz = linearize_pz()
    # print(np.shape(negt2pz))
    adcpz = np.vstack((posadcpz, negadcpz))
    # combine t2wi and adc, respectively
    t2pz = np.vstack((post2pz, negt2pz))

    # shuffle rows respectively
    adcpz, t2pz, clinsig_vect = shuffle(adcpz, t2pz, clinsig_vect)

    np.save('test_adc_as.npy', adcpz)
    np.save('test_t2_as.npy', t2pz)
    np.save('test_clinsig_vect_as.npy', clinsig_vect)

    print('finished')