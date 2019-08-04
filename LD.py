#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:00:08 2019

@author: r2d2
"""

import numpy as np
from scipy.special import *

import math
import warnings
import _variables as dat

def LD(wl,material):
    '''
%***********************************************************************
%   DESCRIPTION:
%   This function computes the complex dielectric constant (i.e. relative
%   permittivity) of various metals using either the Lorentz-Drude (LD) or
%   the Drude model (D). The LD model is the default choice since it 
%   provides a better fit with the exact values. The dielectric function of
%   pure water is calculated with a 2-pole Debye model valid for microwave 
%   frequencies and a 5-pole Lorentz model valid for higher frequencies. 
%   
%   Reference [1] should be used to cite this Matlab code. 
%
%   The Drude-Lorentz parameters for metals are taken from [2] while the 
%   Debye-Lorentz parameters for pure water are from [3].
%   
%***********************************************************************
%   Program author:    Bora Ung
%                      Ecole Polytechnique de Montreal
%                      Dept. Engineering physics
%                      2500 Chemin de Polytechnique
%                      Montreal, Canada
%                      H3T 1J4
%                      boraung@gmail.com

    lambda   ==> wavelength (meters) of light excitation on material.
                    Accepts either vector or matrix inputs.
 
       material ==>    'Ag'  = silver
                       'Al'  = aluminum
                       'Au'  = gold
                       'Cu'  = copper
                       'Cr'  = chromium
                       'Ni'  = nickel
                       'W'   = tungsten
                       'Ti'  = titanium
                       'Be'  = beryllium
                       'Pd'  = palladium
                       'Pt'  = platinum
%   REFERENCES:
%
%   [1] B. Ung and Y. Sheng, Interference of surface waves in a metallic
%       nanoslit, Optics Express (2007)
%   [2] Rakic et al., Optical properties of metallic films for vertical-
%       cavity optoelectronic devices, Applied Optics (1998)
%   [3] J. E. K. Laurens and K. E. Oughstun, Electromagnetic impulse,
%       response of triply distilled water, Ultra-Wideband /
%       Short-Pulse Electromagnetics (1999)
    '''
    wl=np.array(wl)*1e-9
    twopic = 1.883651567308853e+09      # twopic=2*pi*c where c is speed of light
    omegalight = []                     # angular frequency of light (rad/s)
    invsqrt2 = 0.707106781186547        # 1/sqrt(2)
    ehbar = 1.519250349719305e+15       # e/hbar where hbar=h/(2*pi) and e=1.6e-19
    
    for i in range(len(wl)):
        omegalight=np.append(omegalight,twopic/wl[i])
        
    if material == 'Ag':
        # Plasma frequency
        omegap = 9.01*ehbar
        # Oscillators' strenght
        f =     np.array([0.845, 0.065, 0.124, 0.011, 0.840, 5.646])
        # Damping frequency of each oscillator
        Gamma = np.array([0.048, 3.886, 0.452, 0.065, 0.916, 2.419])*ehbar
        # Resonant frequency of each oscillator
        omega = np.array([0.000, 0.816, 4.481, 8.185, 9.083, 20.29])*ehbar
        # Number of resonances
        order = len(omega)
    elif material == 'Au':
        omegap = 9.03*ehbar
        f =     np.array([0.760, 0.024, 0.010, 0.071, 0.601, 4.384])
        Gamma = np.array([0.053, 0.241, 0.345, 0.870, 2.494, 2.214])*ehbar
        omega = np.array([0.000, 0.415, 0.830, 2.969, 4.304, 13.32])*ehbar
        order = len(omega) 
    elif material == 'Ti':
        omegap = 7.29*ehbar
        f =     np.array([0.148, 0.899, 0.393, 0.187, 0.001])
        Gamma = np.array([0.082, 2.276, 2.518, 1.663, 1.762])*ehbar
        omega = np.array([0.000, 0.777, 1.545, 2.509, 1.943])*ehbar
        order = len(omega)
    elif material == 'Cu':
        omegap = 10.83*ehbar
        f =     np.array([0.575, 0.061, 0.104, 0.723, 0.638])
        Gamma = np.array([0.030, 0.378, 1.056, 3.213, 4.305])*ehbar
        omega = np.array([0.000, 0.291, 2.957, 5.300, 11.18])*ehbar
        order = len(omega)
    elif material == 'Cr':
        omegap = 10.75*ehbar; 
        f =     np.array([0.168, 0.151, 0.150, 1.149, 0.825]);
        Gamma = np.array([0.047, 3.175, 1.305, 2.676, 1.335])*ehbar
        omega = np.array([0.000, 0.121, 0.543, 1.970, 8.775])*ehbar
        order = len(omega)
        
    epsilon_D = np.ones(len(wl)) - ((f[0]*omegap**2)*(omegalight**2 + 1j*Gamma[0]*omegalight)**(-1))
    #print epsilon_D
    
    epsilon_L = np.zeros(len(wl));
    for k in range(1,order):
        epsilon_L = epsilon_L + (f[k]*omegap**2)/(((omega[k]**2)*np.ones(len(wl)) - omegalight**2) -
                1j*Gamma[k]*omegalight)
    
    epsilon = epsilon_D + epsilon_L
    
    N = invsqrt2*(((epsilon.real**2+epsilon.imag**2)**(0.5) + epsilon.real)**(0.5) -
        1j*((epsilon.real**2+epsilon.imag**2)**(0.5) - epsilon.real)**(0.5))

    return N
    
    

