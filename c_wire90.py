#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 20:11:28 2019

@author: r2d2
"""
import numpy as np
from scipy.special import *
from LD import LD
import math
import warnings

def C_wire90(wl,material,nout,rad,v,pol):
    '''
    function for computing optical normalised cross sections of a wire
    the k-vector of the incident wave is perpendicular to the wire axis
    the equation taken from Bohren at al. - Adsorption and scattering of light by small particles
    Input:
    
    wl      list of the wavelengths in [nm]
    material ==>    'Ag'  = silver
                    'Au'  = gold
                    'Cu'  = copper
                    'Cr'  = chromium
                    'Ti'  = titanium
                    
    nout    refractive index of the surrounding medium
    rad     radius in [nm] of the wire
    v       integer order of the expansion
    pol     'TM' or 'TE' - polarisation of the EM wave
            'TE' E-field perpendicular to the wire axes    
    '''
    print "wavelenght: " + str(type(wl))
    print "refractive index medium: "+ str(nout)
    print "radius of wire in nm: "+ str(rad)
    print "order of the expansion: "+ str(v)
    print "Polarisation: "+ str(pol)
    
    wl=list(wl)
    
    n=np.conj(LD(wl,material))
    
    n=list(n)
    
    if len(wl) != len(n):
        warnings.warn("The size of the wavelenght list and refractive index list must be the same",Warning)
    else:
        
        x=[]        #size parameter
        m=[]        #relative refractive index
        mx=[]
        cn = np.zeros((len(n),v),dtype=np.complex_)
        
        for i in range(len(wl)):
            x=np.append(x, 2*math.pi*nout*rad/wl[i])
            m=np.append(m, n[i]/nout)
            mx=np.append(mx, m[i]*x[i])
        
        if pol == 'TM': #cn = an coefficients
            for k in range(v):
                cn[0:,k]=(m*jvp(k,x)*jv(k,mx)-jvp(k,mx)*jv(k,x))/(m*jv(k,mx)*h1vp(k,x)-jvp(k,mx)*hankel1(k,x))
            print 'TM polarisation'
            
        elif pol == 'TE':#cn = bn coefficients
            for k in range(v):
                cn[0:,k]=(jvp(k,x)*jv(k,mx)-m*jvp(k,mx)*jv(k,x))/(jv(k,mx)*h1vp(k,x)-m*jvp(k,mx)*hankel1(k,x))
            print 'TE polarisation'
        else:
            warnings.warn("Warning, udefinied polarisation",Warning)
        
        cx=np.matrix(cn)
        x= np.array(x)[np.newaxis]
        
        mat = np.matrix(np.zeros((len(n),4)))

        mat[0:,0] = np.matrix(wl).T/1000 #Wavelenght range
        mat[0:,1] = np.matrix((2*cx.real.sum(axis=1)-cx.real[0:,0])*2/x.T)*2*rad #extincion 
        mat[0:,2] = np.matrix(((2*abs(np.square(cx))).sum(axis=1)-(abs(np.square(cx[0:,0]))))*2/x.T)*2*rad #scattering
        mat[0:,3] = mat[0:,1]-mat[0:,2]  #absorbtion
        
    return mat

