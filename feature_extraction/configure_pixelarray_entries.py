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
    post2pz = np.load('post2pz.npy')
    posadcpz = np.load('posadcpz.npy')

    # make unrandomized clinsig vect
    clinsig_vect = np.vstack((np.ones((175, 1)), np.zeros((560, 1))))

    # linearize negative pixel arrays
    negadcpz, negt2pz = linearize_pz()

    # combine t2wi and adc, respectively
    adcpz = np.vstack((posadcpz, negadcpz))
    t2pz = np.vstack((post2pz, negt2pz))

    # shuffle rows respectively
    adcpz, t2pz, clinsig_vect = shuffle(adcpz, t2pz, clinsig_vect)

    # np.save('adcpz.npy', adcpz)
    # np.save('t2pz.npy', t2pz)
    # np.save('clinsig_vect_pz.npy', clinsig_vect)