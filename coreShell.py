#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:27:19 2019

@author: Daniel Penagos
"""

import numpy as np
from scipy.special import *
from LD import LD
import h5py

import math
import warnings

def bj(v,x):
    #'besselj(n+0.5,x).*sqrt(pi./x/2)','n','x'
    n=np.multiply(jv(v+1.5,x),np.sqrt(np.divide(math.pi,x)/2))
    return n
def by(v,x):
    #'bessely(n+0.5,x).*sqrt(pi./x/2)','n','x'
    n=np.multiply(yv(v+1.5,x),np.sqrt(np.divide(math.pi,x)/2))
    return n
def bh(v,x):
    #'besselh(n+0.5,x).*sqrt(pi./x/2)','n','x'
    n=np.multiply(hankel1(v+1.5,x),np.sqrt(np.divide(math.pi,x)/2))
    return n
def rfx(v,x):
    #inline('sqrt(pi/8.*x).*(besselj(n+0.5,x)./x + besselj(n-0.5,x) - besselj(n+1.5,x))','n','x');
    n=np.multiply(np.sqrt(np.multiply(math.pi,x)/8),np.divide(jv(v+1.5,x),x)+jv(v+0.5,x)-jv(v+2.5,x))
    return n
def rcx(v,x):
    #inline('sqrt(pi/8.*x).*(-bessely(n+0.5,x)./x - bessely(n-0.5,x) + bessely(n+1.5,x))','n','x');
    n=np.multiply(np.sqrt(np.multiply(math.pi,x)/8),-np.divide(yv(v+1.5,x),x)-yv(v+0.5,x)+yv(v+2.5,x))
    return n
def rkx(v,x):
    #inline('sqrt(pi/8.*x).*(besselh(n+0.5,x)./x + besselh(n-0.5,x) - besselh(n+1.5,x))','n','x');
    n=np.multiply(np.sqrt(np.multiply(math.pi,x)/8),np.divide(hankel1(v+1.5,x),x)+hankel1(v+0.5,x)-hankel1(v+2.5,x))
    return n
def rf(v,x):
    #x.*besselj(n+0.5,x).*sqrt(pi./x/2)
    n=np.multiply(np.multiply(x,jv(v+1.5,x)),np.sqrt(np.divide(math.pi,x)/2))
    return n
def rk(v,x):
    #rk = inline('x.*besselh(n+0.5,x).*sqrt(pi./x/2)','n','x');
    n=np.multiply(np.multiply(x,hankel1(v+1.5,x)),np.sqrt(np.divide(math.pi,x)/2))
    return n
def rc(v,x):
    #rc = inline('-x.*bessely(n+0.5,x).*sqrt(pi./x/2)','n','x');
    n=np.multiply(np.multiply(-x,yv(v+1.5,x)),np.sqrt(np.divide(math.pi,x)/2))
    return n

def c_coreshell(wl,mat_core,mat_shell,Nout,rad_core,rad_shell,v):
    '''
    % functin for computing cross sections of core/shell nanoparticles -
    % normalised (Q)
    % the equation taken from Bohren at al. - Adsorption and scattering of light by small particles
    % Input:
        
    wl      list of the wavelengths in [nm]
    materials==>    'Ag'  = silver
                    'Au'  = gold
                    'Cu'  = copper
                    'Cr'  = chromium
                    'Ti'  = titanium
                    
    Nout    refractive index of the surrounding medium
    
    % 	ref ... matrix [wavelength n+i*k(core) n+ik(shell)]   in [nm]
    % 	Nout ... refractive index of medium
    % 	a ... radius of the nanoparticle in nm
    % 	shell ... thickness of a  shell in nm
    % 	n ... order
    % Output:
    % 	mat ... matrix [wavelength extinction scattering absorption Escattered^2(r-comp on surface) (Escattered+Eincident)^2(r-comp on surface)] 
    % Example: - Extinction of silicon sphere 
    %	w = (700:1400)';
    %	mat = C_coreshell([w 3.7*ones(size(w)) 3.7*ones(size(w))],1,140,10,3);
    %	figure; plot(mat(:,1),mat(:,2));
    % modified: 2015-10-08 Stranik
    
    
    % x,y ... size parameters
    % m1,m2 ... rel. refractive indexes
    % bj, by, bh ... spherical bessel 1.kind and third kind
    % rfx(z), rkx(z), rcx(z) ... =(rf)',  =(rk)',  =(rc)'  
    % rf,rk, rc ... riccati-bessel function ..=z*bj(z),  =z*bh(z),  =-z*by(z)
    '''
    
    #refractive index of the shell
    Nshell= np.conj(LD(wl,mat_shell))
    #refractive index of the core
    Ncore= np.conj(LD(wl,mat_core))
    x=np.array([])
    y=np.array([])
    m1=np.array([])
    m2=np.array([])
    
    for i in range(len(wl)):
        x=np.append(x, 2*math.pi*Nout*rad_core/wl[i])
        y=np.append(y, 2*math.pi*Nout*(rad_core+rad_shell)/wl[i])
        m1=np.append(m1, Ncore[i]/Nout)
        m2=np.append(m2, Nshell[i]/Nout)
   
    m1x = np.multiply(m1,x)
    m2x = np.multiply(m2,x)
    m1y = np.multiply(m1,y)
    m2y = np.multiply(m2,y)
    
    an = np.zeros((len(wl),v),dtype=np.complex_)
    bn = np.zeros((len(wl),v),dtype=np.complex_)
    An = np.zeros((len(wl),v),dtype=np.complex_)
    Bn = np.zeros((len(wl),v),dtype=np.complex_)
    hnsu = np.zeros((len(wl),v),dtype=np.complex_)
    jnsu = np.zeros((len(wl),v),dtype=np.complex_)
    jnynsu = np.zeros((len(wl),v),dtype=np.complex_)

    
    for k in range(v):
        
        An[0:,k]=np.divide((np.multiply(np.multiply(m2,rf(k,m2x)),rfx(k,m1x))-np.multiply(np.multiply(m1,rfx(k,m2x)),rf(k,m1x))),
                           (np.multiply(np.multiply(m2,rc(k,m2x)),rfx(k,m1x))-np.multiply(np.multiply(m1,rcx(k,m2x)),rf(k,m1x))))
        Bn[0:,k]=np.divide((np.multiply(np.multiply(m2,rf(k,m1x)),rfx(k,m2x))-np.multiply(np.multiply(m1,rf(k,m2x)),rfx(k,m1x))),
                           (np.multiply(np.multiply(m2,rcx(k,m2x)),rf(k,m1x))-np.multiply(np.multiply(m1,rfx(k,m1x)),rc(k,m2x))))
        an[0:,k]=np.divide((np.multiply(rf(k,y),rfx(k,m2y))-np.multiply(An[:,k],rcx(k,m2y))-np.multiply(np.multiply(m2,rfx(k,y)),rf(k,m2y))-np.multiply(An[:,k],rc(k,m2y))),
                           (np.multiply(rk(k,y),rfx(k,m2y))-np.multiply(An[:,k],rcx(k,m2y))-np.multiply(np.multiply(m2,rkx(k,y)),rf(k,m2y))-np.multiply(An[:,k],rc(k,m2y))))
        bn[0:,k]=np.divide((np.multiply(np.multiply(m2,rf(k,y)),rfx(k,m2y))-np.multiply(Bn[:,k],rcx(k,m2y))-np.multiply(rfx(k,y),rf(k,m2y))-np.multiply(Bn[:,k],rc(k,m2y))),
                           (np.multiply(np.multiply(m2,rk(k,y)),rfx(k,m2y))-np.multiply(Bn[:,k],rcx(k,m2y))-np.multiply(rkx(k,y),rf(k,m2y))-np.multiply(Bn[:,k],rc(k,m2y))))
        hnsu[0:,k]=np.square(abs(bh(k,y)))
        jnsu[0:,k]=np.square(bj(k,y))
        jnynsu[0:,k]=np.multiply(bj(k,y),by(k,y))
    
    vect=np.linspace(1,v,v)
    vect1=np.multiply(vect,2)+1
    multip1=np.ones((len(wl),1))*vect1

    mat = np.matrix(np.zeros((len(wl),6)))
    
    mat[0:,0] = np.matrix(wl).T/1000 #Wavelenght range
    mat[0:,1] = np.matrix(np.multiply(np.square(np.divide(1,y))*2,np.multiply(multip1,(an+bn).real).sum(axis=1))).T #extincion
    mat[0:,2] = np.matrix(np.multiply(np.square(np.divide(1,y))*2,(np.multiply(multip1,np.square(abs(an)))+np.square(abs(bn))).sum(axis=1))).T #scattering
    mat[0:,3] = mat[0:,1]-mat[0:,2]  #absorbtion

    vect2=np.multiply(np.multiply(np.multiply(vect,2)+1,vect+1),vect)
    multip2=np.ones((len(wl),1))*vect2
    #Es^2 r-comp
    mat[0:,4] = np.matrix(np.multiply(np.square(np.divide(1,y))*1.5,np.multiply(np.multiply(multip2,hnsu),np.square(abs(an))).sum(axis=1))).T 
    #(Es+Ei)^2, r-comp
    mat[0:,5] = np.matrix(np.multiply((np.square(np.divide(1,y))*1.5),(np.multiply(multip2,(np.multiply(hnsu,np.square(abs(an)))-2*np.multiply(an.real,jnsu)+2*np.multiply(an.imag,jnynsu))).sum(axis=1)))+1).T
   
    return mat

hf = h5py.File('data.h5', 'r')
wl = hf.get('wl')
wl = np.array(wl)
hf.close()

x=c_coreshell(wl*1000,'Ag','Ag',1,15,10,5)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x[0:,0],x[0:,1], '-g',label='extintion')
plt.plot(x[0:,0],x[0:,2], '-b',label='scattering')
plt.plot(x[0:,0],x[0:,3], '-r',label='absorbtion')

plt.xlabel("wavelength (um)")
plt.ylabel("Efficiencies")
#plt.legend(loc="upper right")  
plt.axis([0.24, 0.7, 0, max(x[0:,1])*1.2])
plt.grid(True)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    