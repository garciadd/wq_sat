"""
Author: Daniel García Díaz
Institute of Physics of Cantabria (IFCA)
Advanced Computing and e-Science
Date: Sep 2018
"""

import numpy as np
import skimage

#Coeficiente de correlacion
def coef_correlation(ms, f):
    
    x = ms.flatten()
    y = f.flatten()
    
    n = len(x)
    sgx = np.sqrt(np.nansum((x - np.nanmean(x))**2)/(n-1))
    sgy = np.sqrt(np.nansum((y - np.nanmean(y))**2)/(n-1))
    sg = np.nansum(((x - np.nanmean(x))*(y - np.nanmean(y))))/(n-1)
    cc = sg / (sgx * sgy)

    return round(cc, 3)

def rmse(ms, f):
    
    x = ms.flatten()
    y = f.flatten()
    
    n = len(x)
    dif_sq = (y - x)**2
    rmse = np.sqrt(np.sum(dif_sq) / n)

    return round(rmse, 3)

def zhou_coeficient(f, P):
    
    #Aplicamos el filtro de laplace como filtro de paso alto
    ksize = 3
    x = skimage.filters.laplace(P, ksize)
    y = skimage.filters.laplace(f, ksize)
    
    xf = x.flatten()
    yf = y.flatten()
    
    #indice Zhou
    zhou = coef_correlation(xf, yf)
    
    return round(zhou, 3)

def espectral_ergas(ms, f):
    
    x = ms.flatten()
    y = f.flatten()
        
    #rmse (root mean squared error)
    #rmse = np.sqrt(mean_squared_error(x, y))
    mse = rmse(x, y)
    erg = 100 * (10/20) * np.sqrt(mse**2 / (np.mean(x))**2)
    #erg = 100 * (h/l) * np.sqrt((np.sum((rmse**2) / np.mean(x)**2)) / b)
    
    return round(erg, 3)

def espatial_ergas(f, P):
    
    #Igualar histogramas: misma media y desviacion estandar
    a = np.nanstd(f) / np.nanstd(P)
    b = np.nanmean(f) - (np.nanstd(f) / np.nanstd(P)) * np.nanmean(P)
    Pe = a * P + b
    
    x = Pe.flatten()
    y = f.flatten()
    
    #rmse (root mean squared error)
    #rmse = np.sqrt(mean_squared_error(x, y))
    mse = rmse(x, y)
    erg = 100 * (10/20) * np.sqrt(mse**2 / (np.mean(x))**2)
    
    return round(erg, 3)

def q_index(ms, f):
    
    x = ms.flatten()
    y = f.flatten()
    n = len(x)
    
    sgxy = (1/(n-1)) * np.sum((x - np.mean(x))*(y - np.mean(y)))
    sgx = np.sqrt((1/(n-1)) * np.sum((x - np.mean(x))**2))
    sgy = np.sqrt((1/(n-1)) * np.sum((y - np.mean(y))**2))
    
    q = (sgxy / (sgx*sgy)) * ((2 * np.mean(x) * np.mean(y)) / (np.mean(x)**2 + np.mean(y)**2)) * ((2 * sgx * sgy) / (sgx**2 + sgy**2))            
                    
    return round(q, 3)