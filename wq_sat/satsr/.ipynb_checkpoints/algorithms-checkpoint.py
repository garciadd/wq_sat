"""
Author: Daniel García Díaz
Institute of Physics of Cantabria (IFCA)
Advanced Computing and e-Science
Date: Sep 2018
""""

import numpy as np
import cv2
import skimage
import pywt

"""
Algoritmo de fusión IHS
------------------------

Se trata de un algoritmo de sustitución.

Para realizar el proceso de fusión IHS, tres bandas MS son asignadas a los canales RGB y estos canales RGB se transforman a sus componentes intensidad-matiz-saturación. Debido a que el canal de intensidad representa el brillo de estas imágenes, se puede sustituir por la banda pancromática de alta resolución.

El histograma del canal de intensidad y el pancromático son diferentes y es necesario realizar la igualación de ambos histogramas con la idea de tener la misma media y desviación estándar.

Esta nueva imagen pancromática remplaza el componente de intensidad; por ultimo, se realiza la transformación inversa de IHS para obtener los nuevos canales RGB.

Ecuaciones de transformación.

#matriz de transformacion
m = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],[1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)], [1/np.sqrt(2), -1/np.sqrt(2), 0]]).astype('float32')

#Matriz de transformacion y transformacion IHS -> RGB con las nuevas componentes
t = np.array([[1/np.sqrt(3), 1/np.sqrt(6), 1/np.sqrt(2)],[1/np.sqrt(3), 1/np.sqrt(6), -1/np.sqrt(2)], [1/np.sqrt(3), -2/np.sqrt(6), 0]]).astype('float32')
"""

def ihs(Rband, Gband, Bband, P):
    
    #matriz de transformacion
    m = np.array([[1/3, 1/3, 1/3],
                  [-np.sqrt(2)/6, -np.sqrt(2)/6, (2*np.sqrt(2))/6], 
                  [1/np.sqrt(2), -1/np.sqrt(2), 0]]).astype('float32')
    
    #Transformacion RGB -> IHS
    I = (m[0][0] * Rband) + (m[0][1] * Gband) + (m[0][2] * Bband)
    V1 = (m[1][0] * Rband) + (m[1][1] * Gband) + (m[1][2] * Bband)
    V2 = (m[2][0] * Rband) + (m[2][1] * Gband) + (m[2][2] * Bband)
    
    #Igualar histogramas: misma media y desviacion estandar
    a = np.nanstd(I) / np.nanstd(P)
    b = np.nanmean(I) - (np.nanstd(I) / np.nanstd(P)) * np.nanmean(P)
    Pe = a * P + b
    
    #Comprobacion
    print ("I Media y desviacion: {}; {}".format(np.nanmean(I), np.nanstd(I)))
    print ("P Media y desviacion: {}; {}".format(np.nanmean(Pe), np.nanstd(Pe)))

    #Matriz de transformacion
    t = np.array([[1, -1/np.sqrt(2), 1/np.sqrt(2)],
                  [1, -1/np.sqrt(2), -1/np.sqrt(2)],
                  [1, np.sqrt(2), 0]]).astype('float32')
    
    # Transformacion IHS -> RGB con las nuevas componentes
    Rf = (t[0][0] * Pe) + (t[0][1] * V1) + (t[0][2] * V2)
    Gf = (t[1][0] * Pe) + (t[1][1] * V1) + (t[1][2] * V2)
    Bf = (t[2][0] * Pe) + (t[2][1] * V1) + (t[2][2] * V2)
        
    return [Rf, Gf, Bf]

""""
Algoritmo de fusión de Brovey
------------------------------
Métodos basados en operaciones algebraicas.

Generalmente se aplica sobre una composición rojo, verde y azul (RGB, Red- Green-Blue) de las bandas espectrales. En una primera etapa, normaliza cada banda o, en su caso, la imagen compuesta, y luego se multiplica el resultado por los datos deseados, en este caso los de la imagen PAN. De esta forma se añade la componente intensidad o brillo de la imagen PAN.

Es facilmente extendible a n bandas aplicando formula a cada una de las bandas.

NDfus = (N * NDbi / NDb1 + NDb2 + ... + NDbn) * NDpan

Ventajas: Fácilmente aplicable. Bajo coste computacional. Imágenes de alta calidad espacial.
Desventaja: Imágenes de baja calidad espectral.
"""

def brovey(ms_bands, P):
        
    #Igualar histogramas: misma media y desviacion estandar
    a = np.nanstd(ms_bands) / np.nanstd(P)
    b = np.nanmean(ms_bands) - (np.nanstd(ms_bands) / np.nanstd(P)) * np.nanmean(P)
    Pe = a * P + b
    
    #Comprobacion
    print ("check if the histograms are equal")
    print ("I Mean y std: {}; {}".format(np.nanmean(ms_bands), np.nanstd(ms_bands)))
    print ("P Mean y std: {}; {}".format(np.nanmean(Pe), np.nanstd(Pe)))
    
    sr_bands = np.zeros(ms_bands.shape)
    for i in range(ms_bands.shape[-1]):
        
        #fusion imagenes algoritmo brovey (N * DNbi / DNb1 + DNb2 + ... + DNn) * P
        arr = ((ms_bands.shape[-1] * ms_bands[:,:, i]) /  np.nansum(ms_bands, axis=(2))) * Pe
        sr_bands[:,:,i] = arr
    
    return sr_bands

"""
Algoritmo de fusion  HPF (High pass filter)
--------------------------------
"""

def hpf(ms_bands, P, uf):
    
    # El tamaño y el valor central del filtro, se tiene que calcular en función de R
    # M tambien depende de R
    if 1 < uf < 2.5:
        ksize=5
        M = 0.25
    elif 2.5 < uf < 3.5:
        ksize=7
        M = 0.50
    elif 3.5 < uf < 5.5:
        ksize=9
        M = 0.50
    elif 5.5 < uf < 7.5:
        ksize=11
        M = 0.65
    
    sr_bands = np.zeros(ms_bands.shape)
    for i in range(ms_bands.shape[-1]):
        
        ms_arr = ms_bands[:,:,i]
    
        #Aplicamos el filtro de laplace como filtro de paso alto
        Pf = skimage.filters.laplace(P, ksize) 

        # factor de ponderación y calculo de cada banda fusionada
        W = (np.nanstd(ms_arr) / np.nanstd(Pf)) * M
        msf = ms_arr + (Pf * W)

        #Igualar histogramas: misma media y desviacion estandar
        a = np.nanstd(ms_arr) / np.nanstd(msf)
        b = np.nanmean(ms_arr) - (np.nanstd(ms_arr) / np.nanstd(msf)) * np.nanmean(msf)
        msfh = a * msf + b

        #Comprobacion
        print ("MS original Media y desviacion: {}; {}".format(np.nanmean(ms_arr), np.nanstd(ms_arr)))
        print ("MS fusionada Media y desviacion: {}; {}".format(np.nanmean(msfh), np.nanstd(msfh)))
        
        sr_bands[:,:,i] = msfh
    
    return sr_bands

"""
Algoritmo de fusion Transformada de Wavelet, metodo de mallat IHS
------------------------------------------------------------------

Paso 1. Resize una composición a color RGB (verdadero color) de la imagen multiespectral con la imagen pancromática, usando el mismo tamaño de píxel de esta última. 

Paso 2. Transformar la imagen RGB en componentes IHS (intensidad, matiz y saturación). 

Paso 3. Aplicar el concepto de TRWH al componente I, iterativamente hasta el segundo nivel descomposición, obteniendo de esta manera los siguientes coeficientes de aproximación y detalle. cA1i coeficientes de aproximación que contienen la información espectral de la componente I, cV1i, cH1i y cD1i coeficientes de detalle donde se almacena la información espacial de I. cA1i se descompone por segunda vez, con el fin de obtienen los coeficientes de aproximación cA2i que contienen la información espectral de I. cV2i, cH2i y cD2i, junto con cV1i, cH1i y cD1i son los coeficientes de detalle donde está almacenada la información espacial de la componente I.

Paso 4. Aplicar el concepto de la TRHW a la imagen pancromática hasta el segundo nivel descomposición obteniendo de esta manera los coeficientes de aproximación y detalle. cA1p coeficientes de aproximación que contiene la información espectral de la imagen pancromática, cV1p, cH1p y cD1p coeficientes de detalle donde se almacena la información espacial de la imagen pancromática. cA1p se descompone por segunda vez obteniendo cA2p que corresponde a los coeficientes de aproximación de segundo nivel, los cuales contienen la información espectral de la imagen pancromática. cV2p, cH2p y cD2p junto con cV1p, cH1p y cD1p son los coeficientes de detalle donde está almacenada la información espacial de la imagen pancromática

Paso 5. Generar una nueva matriz concatenando los coeficientes cA2i (que almacena la información de la componente I) y los coeficientes de detalle de segundo nivel de la imagen pancromática cV2p, cH2p ycD2p, junto con los coeficientes de detalle de la descomposición de primer nivel cV1p, cH1p y cD1p. 

Paso 6. Aplicar la transformada inversa de la TRHW (I-TRWH) a la matriz obtenida en el paso anterior para obtener la nueva componente intensidad (N-INT). 

Paso 7. Generar una nueva composición IHS (N-IHS), uniendo la N-INT (nuevo componente intensidad) junto con las componentes originales de matiz y saturación (obtenidas en el paso 2). 

Paso 8. Realizar la transformación IHS a RGB, usando la nueva composición N-IHS. De esta manera se obtiene la nueva imagen multiespectral, que mantiene la resolución espectral ganando así la resolución espacial.

"""

def trwh_ihs(Rband, Gband, Bband, P):

    #matriz de transformacion
    m = np.array([[1/3, 1/3, 1/3],
                  [-np.sqrt(2)/6, -np.sqrt(2)/6, (2*np.sqrt(2))/6], 
                  [1/np.sqrt(2), -1/np.sqrt(2), 0]]).astype('float32')
    
    #Transformar la imagen RGB en componentes IHS
    I = (m[0][0] * Rband) + (m[0][1] * Gband) + (m[0][2] * Bband)
    H = (m[1][0] * Rband) + (m[1][1] * Gband) + (m[1][2] * Bband)
    S = (m[2][0] * Rband) + (m[2][1] * Gband) + (m[2][2] * Bband)
    
    #Igualar histogramas: misma media y desviacion estandar
    a = np.nanstd(I) / np.nanstd(P)
    b = np.nanmean(I) - (np.nanstd(I) / np.nanstd(P)) * np.nanmean(P)
    Pe = a * P + b
    
    #Comprobacion
    print ("I Media y desviacion: {}; {}".format(np.nanmean(I), np.nanstd(I)))
    print ("P Media y desviacion: {}; {}".format(np.nanmean(Pe), np.nanstd(Pe)))
    
    #Aplicar TRWH a la componente I hasta el segundo nivel descomposición
    coeffsI = pywt.wavedec2(I, 'db4', level=2)
    #Aplicar TRWH a la componente P hasta el segundo nivel descomposición
    coeffsPe = pywt.wavedec2(Pe, 'db4', level=2)
    
    #Nueva matriz concatenando los coeficientes cA2i, cV2p, cH2p ycD2p, cV1p, cH1p y cD1p. 
    new_coeffs = coeffsPe
    new_coeffs[0] = coeffsI[0]
    #transformada inversa de la TRHW
    IP = pywt.waverec2(new_coeffs, 'db4')

    #Matriz de transformacion
    t = np.array([[1, -1/np.sqrt(2), 1/np.sqrt(2)],
                  [1, -1/np.sqrt(2), -1/np.sqrt(2)],
                  [1, np.sqrt(2), 0]]).astype('float32')

    #Reconstruccion de las bandas con la nueva matriz IP
    Rf = (t[0][0] * IP) + (t[0][1] * H) + (t[0][2] * S)    
    Gf = (t[1][0] * IP) + (t[1][1] * H) + (t[1][2] * S)
    Bf = (t[2][0] * IP) + (t[2][1] * H) + (t[2][2] * S)
    
    return [Rf, Gf, Bf]