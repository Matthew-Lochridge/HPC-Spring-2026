import numpy as np

# Kurokawa pseudopotentials

def V_C(q, config):
    '''
    Pseusopotential function for carbon atoms (Kurokawa)
    '''
    return (config.a_d/2)**3 * (config.b_C[0] * (config.b_C[2]*q**2 - config.b_C[1]) / (np.exp(config.b_C[2]*q**2 - config.b_C[3]) + 1))

def V_H(q, config):
    '''
    Pseudopotential function for hydrogen atoms (Kurokawa)
    '''
    return (config.a_d/2)**3 * ((q <= 2) * (config.b_H[0] + config.b_H[1] * q + config.b_H[2] * q**2 + config.b_H[3] * q**3) + (q > 2) * (config.b_H[4] / (q + (q == 0)) + config.b_H[5] / (q**2 + (q == 0)) + config.b_H[6] / (q**3 + (q == 0)) + config.b_H[7] / (q**4 + (q == 0))))
