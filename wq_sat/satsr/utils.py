import numpy as np
import cv2

def data_resize(data_bands):
    
    max_res = np.amin(list(data_bands.keys()))
    m, n = data_bands[max_res].shape[:2]
    
    resolutions = [res for res in list(data_bands.keys()) if res != max_res]
    
    rs_bands = {}    
    for res in resolutions:
        arr_bands =  np.zeros((m, n, data_bands[res].shape[-1]))
        for i in range(data_bands[res].shape[-1]):
            arr_bands[:,:,i] = cv2.resize(data_bands[res][:,:,i], (n, m) , interpolation=cv2.INTER_CUBIC)
        rs_bands[res] = arr_bands
    
    return rs_bands