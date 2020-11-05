"""
Author: Daniel García Díaz
Institute of Physics of Cantabria (IFCA)
Advanced Computing and e-Science
Date: Sep 2018
"""

import numpy as np

# Subfunctions
import utils

def load_satellite_module(selected_sat):
    print('Loading satellite functions for {}...'.format(selected_sat))
    
    global satellite

    if selected_sat == 'sentinel':
        import wq_sat.satellites.sentinel as satellite

    elif selected_sat == 'landsat':
        import wq_sat.satellites.landsat as satellite
    
    else:
        raise Exception('Invalid satellite name')

def load_fusion_alg(method):
    print('Loading {} fusion algorithm ...'.format(method))
    
    global fus_alg
    
    if method == 'ihs':
        import algorithms.ihs as fus_alg
    elif method == 'brovey':
        import algorithms.brovey as fus_alg
    elif method == 'hpf':
        import algorithms.hpf as fus_alg
    elif method == 'trwh_ihs':
        import algorithms.trwh_ihs as fus_alg
    else:
        raise Exception('Invalid fusion algorithm name')

def sr_predict(selected_sat, data_bands, method):
    
    # Load satellite functions
    load_satellite_module(selected_sat)
    
    # Load fusion algorithm
    load_fusion_alg(method)
    
    # Normalize pixel values  and put image in float32 format
    for res in data_bands.keys():
        data_bands[res] = data_bands[res].astype(np.float32)
        data_bands[res] = (data_bands[res] - satellite.min_val) / (satellite.max_val - satellite.min_val)

    # Building the panchromatic band
    max_res = np.amin(list(data_bands.keys()))
    P = np.mean(data_bands[max_res], axis=2)

    resize_bands = utils.data_resize(data_bands)

    sr_bands = {}
    sr_bands[max_res] = data_bands[max_res]

    for res in resize_bands.keys():
        if method ==  'brovey':
            sr_bands[res] = fus_alg(resize_bands[res], P)

        elif method == 'hpf':
            uf = upscaling_factor[res]
            sr_bands[res] = fus_alg(resize_bands[res], P, uf)

        elif method == 'ihs' or method == 'trwh_ihs':
            arr_bands = np.zeros(resize_bands[res].shape)

            R, G, B = resize_bands[res][:,:,0], resize_bands[res][:,:,1], resize_bands[res][:,:,2]
            rgb_list = fus_alg(R, G, B, P)
            arr_bands[:,:,0], arr_bands[:,:,1], arr_bands[:,:,2] = rgb_list[0], rgb_list[1], rgb_list[2] 

            for i in range((resize_bands[res].shape[-1] - 3)):
                i +=3
                R, G, B = resize_bands[res][:,:,0], resize_bands[res][:,:,1], resize_bands[res][:,:,i]
                rgb_list = fus_alg(R, G, B, P)
                arr_bands[:,:,i] = rgb_list[-1]
            sr_bands[res] = arr_bands

    for res in sr_bands.keys():
        sr_bands[res] = sr_bands[res] * (satellite.max_val - satellite.min_val) + satellite.min_val
        sr_bands[res] = np.clip(sr_bands[res], a_min=satellite.min_val, a_max=satellite.max_val)
        
    return sr_bands