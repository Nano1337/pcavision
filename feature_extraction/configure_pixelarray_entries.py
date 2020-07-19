import numpy as np
from sklearn.utils import shuffle
def linearize_pz():
    unconfig_adcpz = np.load('C:\\Users\\haoli\\Documents\\pcavision\\manual_extraction\\adcpz.npy')
    unconfig_t2pz = np.load('C:\\Users\\haoli\\Documents\\pcavision\\manual_extraction\\t2pz.npy')

    #linearize ADC
    config_adcpz = np.zeros((560, 6, 6))
    i = 0
    for r in range(len(unconfig_adcpz)):
        for c in range(len(unconfig_adcpz[r])):
            config_adcpz[i] = unconfig_adcpz[r, c]
            i += 1

    #linearize T2WI
    config_t2pz = np.zeros((560, 16, 16))
    i = 0
    for r in range(len(unconfig_t2pz)):
        for c in range(len(unconfig_t2pz[r])):
            config_t2pz[i] = unconfig_t2pz[r, c]
            i += 1
    return config_adcpz, config_t2pz


if __name__ == "__main__":
    # unrandomized positive on top and negative on bottom
    post2pz = np.load('C:\\Users\\haoli\\Documents\\pcavision\\lesion_extraction_2d\\test_pz_t2.npy')
    posadcpz = np.load('C:\\Users\\haoli\\Documents\\pcavision\\lesion_extraction_2d\\test_pz_adc.npy')

    # make unrandomized clinsig vect
    clinsig_vect = np.vstack((np.ones((113, 1)), np.zeros((560, 1))))

    # linearize negative pixel arrays
    negadcpz, negt2pz = linearize_pz()

    adcpz = np.vstack((posadcpz, negadcpz))
    # combine t2wi and adc, respectively
    t2pz = np.vstack((post2pz, negt2pz))

    # shuffle rows respectively
    adcpz, t2pz, clinsig_vect = shuffle(adcpz, t2pz, clinsig_vect)

    np.save('testalladcpz.npy', adcpz)
    np.save('testallt2pz.npy', t2pz)
    np.save('testallclinsig_vect.npy', clinsig_vect)

    print('finished')