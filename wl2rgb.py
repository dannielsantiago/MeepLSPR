#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:34:21 2019

@author: r2d2
"""

import numpy as np
import matplotlib.pyplot as plt
from coreShell import c_coreshell
import h5py
import c_wire90 as theoretical

def wavelength_to_rgb(wavelength, gamma=255):

    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    #R *= 255
    #G *= 255
    #B *= 255
    return np.array([R, G, B])

def showColor(wl):
    rgb=wavelength_to_rgb(wl)
    ones=rgb2ones(rgb)
    plt.figure()
    plt.imshow(ones.T)
    plt.axis('off')

    
def rgb2ones(rgb):
    N=20
    a=abs(np.linspace(-1,1,N))    
    ones=np.ones((3,N,N))
    ones[0:,0:,:]-=a
    ones[0,:,:]=ones[0,:,:]*ones[0,:,:].T
    ones[1,:,:]=ones[1,:,:]*ones[1,:,:].T
    ones[2,:,:]=ones[2,:,:]*ones[2,:,:].T
    ones[0,:,:]*=rgb[0]
    ones[1,:,:]*=rgb[1]
    ones[2,:,:]*=rgb[2]
    return ones
    
def spectra2RGB(wl,scat):
    scat/=scat.max()       
    rgb=np.array([0,0,0],dtype=float)
    for i in range(len(wl)):
        temp=wavelength_to_rgb(wl[i])
        temp*=float(scat[i])
        rgb+=temp
        
    print(rgb)
    rgb/=rgb.max()
    print(rgb)
    return rgb
    

w=0.43
fcen=1/w
df=1.8
minn=fcen-df/2.0
maxx=fcen+df/2.0
wl=np.linspace(minn,maxx,100)[::-1]
wl=1000/wl
mat=c_coreshell(wl,'Ag','Ag',1,60,0,5)
#mat=theoretical.C_wire90(wl,'Ag',1.0,0.050*1000,16,'TM')
plt.figure()
plt.subplot(1,2,1)
plt.plot(mat[0:,0],mat[0:,3], '-',label='Abso')#scat
plt.plot(mat[0:,0],mat[0:,2], '-',label='Scat')#scat
plt.plot(mat[0:,0],mat[0:,1], '-',label='Ext')#scat

plt.xlabel("wavelength (um)")
plt.ylabel("cross-section (nm)")
plt.legend(loc="upper right",title="") 
plt.axis([0.31, 0.7, 0, mat[0:,1].max()*1.2])
plt.grid(True)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.subplot(1,2,2)
color=rgb2ones(spectra2RGB(wl,mat[0:,2]))
plt.imshow(color.T)
plt.axis('off')


showColor(632.8)