#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:19:54 2019

@author: r2d2
"""

 #Import libraries
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from meep.materials import Au
import _materials as lib
import _shapes as shapes
from scipy.signal import find_peaks



'''
------------------------Parameters of the simulation
'''
rad=0.025
w=0.31                  # wavelength
fcen=1/w                # Pulse center frequency
df = 3                  # pulse width in micrometers

axis=mp.Ex              # Axis of direction of the pulse Ex=TM, Hx=TE
dpml = w                # Width of th pml layers = wavelength

sx = 14*rad             # Size of inner shell
sy = 14*rad             # Size of inner shell
sz = 14*rad             # Size of inner shell

sx0 = sx + 2*dpml       # size of cell in X direction
sy0 = sy + 2*dpml       # size of cell in Y direction
sz0 = sz + 2*dpml       # size of cell in z direction

mx = 8*rad              # size of monitor box side
fx = 4*rad              # size of flux box side

sc=6*rad                # source center coordinate shift
sw=sx0                  # source width, needs to be bigger than inner cell to generate only plane-waves

nfreq = 100             # number of frequencies at which to compute flux
courant=0.25            # numerical stability, default is 0.5, should be lower in case refractive index n<1

time_step=0.1           # time step to measure flux
add_time=10             # additional time until field decays 1e-6
resolution = 200        # resolution pixels/um (pixels/micrometers)
resolutionImages = 200     # resolution pixels/um (pixels/micrometers) to run the images
decay = 1e-3           # decay limit condition for the field measurement


cell = mp.Vector3(sx0, sy0, sz0) 
monitor = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(mx,mx,mx))
monitor2D = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(0,mx,mx))


runImages = True
runData = False

#Inside the geometry object, the device structure is specified together with its center and the type of the material used
geometry = [shapes.sph]

#Boundary conditions using Perfectly Maching Layers PML// ficticius absorbtion material to avoid reflection of the fields
pml_layers = [mp.PML(dpml)]

sym = [mp.Mirror(mp.X, phase=-1)] # symmetry condition to reduce simulation time

'''
2D view of the box

  +--------------------Cell-------------------+
  |                                           |
  |                    PML                    |
  |     +-------------------------------+     |
  |     |                               |     |
  |     |              .pt              |     |
  |     |                               |     |
  |     |     +---+monitor-area---+     |     |
  |     |     |                   |     |     |
  |     |     |   +---Flux----+   |     |     |
  |     |     |   |           |   |     |     |
  |     |     |   |    XXX    |   |     |     |
  |     |     |   |   XXXXX   |   |     |     |
  |     |     |   |    XXX    |   |     |     |
  |     |     |   |           |   |     |     |
  |     |     |   +--region---+   |     |     |
  |     |     |                   |     |     |
  |     |     +-------------------+     |     |
  |     |                               |     |
  +------------------Source-------------------+
  |     |                               |     |
  |     +-------------------------------+     |
  |                                           |
  |                                           |
  +-------------------------------------------+

'''



'''
y              
^   k             
|   ^             
|   |          
|              
+-------> x    
Gaussian       
'''

# Defining the sources
gaussian=mp.Source(mp.GaussianSource(wavelength=w,fwidth=df),
                     component=axis,
                     center=mp.Vector3(0,-sc,0),
                     size=mp.Vector3(sw,0,sw))

pt=mp.Vector3(0,sc,0) # point used to measure decay of Field (in oposite side of the source)

sources = [gaussian]



'''
-------------------------------
Regions for the flux measurement
-------------------------------

                 flux monitor > refl surfaces

                   refl_fr_t
               +---------------+
               |  nanoparticle |
               |      XXXX     |
               |     XXXXXX    |
     refl_fr_l |     XXXXXX    | refl_fr_r
               |      XXXX     |
               |   radius=r    |
               |               |
               +---------------+
                   refl_fr_b
            
               +---------------+
                    tran_fr
                   


'''
# reflected flux / Regions (top,bottom,left,right)
refl_fr_t = mp.FluxRegion(center=mp.Vector3(0,-fx/2,0), size=mp.Vector3(fx,0,fx))
refl_fr_b = mp.FluxRegion(center=mp.Vector3(0,fx/2,0), size=mp.Vector3(fx,0,fx))
refl_fr_l = mp.FluxRegion(center=mp.Vector3(-fx/2,0,0), size=mp.Vector3(0,fx,fx))
refl_fr_r = mp.FluxRegion(center=mp.Vector3(fx/2,0,0), size=mp.Vector3(0,fx,fx))

refl_fr_up = mp.FluxRegion(center=mp.Vector3(0,0,fx/2), size=mp.Vector3(fx,fx,0))
refl_fr_dw = mp.FluxRegion(center=mp.Vector3(0,0,-fx/2), size=mp.Vector3(fx,fx,0))

# transmitted flux, origin at source, but size as large as the flux side
tran_fr = mp.FluxRegion(center=mp.Vector3(0,-sc,0), size=mp.Vector3(fx,0,fx))

if runData:
    print('something')
    '''
    ------------------------------
    1st simulaton without particle and Get normal flux
    ------------------------------
    '''

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=[],
                        sources=sources,
                        symmetries=sym,
                        resolution=resolution,
                        Courant=courant)
    
    refl_t = sim.add_flux(fcen, df, nfreq, refl_fr_t)
    refl_b = sim.add_flux(fcen, df, nfreq, refl_fr_b)
    refl_l = sim.add_flux(fcen, df, nfreq, refl_fr_l)
    refl_r = sim.add_flux(fcen, df, nfreq, refl_fr_r)
    
    refl_up = sim.add_flux(fcen, df, nfreq, refl_fr_up)
    refl_dw = sim.add_flux(fcen, df, nfreq, refl_fr_dw)
    
    
    tran = sim.add_flux(fcen,df,nfreq,tran_fr)
    
    sim.use_output_directory('flux-sph_0')
    sim.run(mp.in_volume(monitor, mp.at_beginning(mp.output_epsilon)),
            mp.in_volume(monitor, mp.to_appended("ez", mp.at_every(time_step, mp.output_efield_x))),
            until_after_sources=mp.stop_when_fields_decayed(add_time,axis,pt,decay))
    
    # for normalization run, save flux fields data for reflection plane
    straight_refl_data_t = sim.get_flux_data(refl_t)
    straight_refl_data_b = sim.get_flux_data(refl_b)
    straight_refl_data_l = sim.get_flux_data(refl_l)
    straight_refl_data_r = sim.get_flux_data(refl_r)
    
    straight_refl_data_up = sim.get_flux_data(refl_up)
    straight_refl_data_dw = sim.get_flux_data(refl_dw)
    
    # save incident power for transmission plane
    incident_tran_flux = mp.get_fluxes(tran)
    flux_freqs = mp.get_flux_freqs(tran)
    
    '''
    ------------------------------------------------
    2nd Simulation with particle and get Scattered Flux
    ------------------------------------------------
    '''
    sim.reset_meep()
    
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        symmetries=sym,
                        resolution=resolution,
                        Courant=courant)
    
    # reflected flux
    refl_t = sim.add_flux(fcen, df, nfreq, refl_fr_t)
    refl_b = sim.add_flux(fcen, df, nfreq, refl_fr_b)
    refl_l = sim.add_flux(fcen, df, nfreq, refl_fr_l)
    refl_r = sim.add_flux(fcen, df, nfreq, refl_fr_r)
    
    refl_up = sim.add_flux(fcen, df, nfreq, refl_fr_up)
    refl_dw = sim.add_flux(fcen, df, nfreq, refl_fr_dw)
    
    # for normal run, load negated fields to subtract incident from refl. fields
    sim.load_minus_flux_data(refl_t, straight_refl_data_t)
    sim.load_minus_flux_data(refl_b, straight_refl_data_b)
    sim.load_minus_flux_data(refl_l, straight_refl_data_l)
    sim.load_minus_flux_data(refl_r, straight_refl_data_r)
    
    sim.load_minus_flux_data(refl_up, straight_refl_data_up)
    sim.load_minus_flux_data(refl_dw, straight_refl_data_dw)
    
    
    sim.use_output_directory('flux-sph_1')
    sim.run(mp.in_volume(monitor, mp.at_beginning(mp.output_epsilon)),
            mp.in_volume(monitor, mp.to_appended("ez", mp.at_every(time_step, mp.output_efield_x))),
            until_after_sources=mp.stop_when_fields_decayed(add_time,axis,pt,decay))
    
    #scat_refl_flux = mp.get_fluxes(refl)                                            
    #save scattered reflected flux from the surfaces
    scat_refl_data_t = mp.get_fluxes(refl_t)
    scat_refl_data_b = mp.get_fluxes(refl_b)
    scat_refl_data_l = mp.get_fluxes(refl_l)
    scat_refl_data_r = mp.get_fluxes(refl_r)
    
    scat_refl_data_up = mp.get_fluxes(refl_up)
    scat_refl_data_dw = mp.get_fluxes(refl_dw)
    
    '''
    --------------------------------
    3rd Simulation with nanoparticle to get the absorbance
    --------------------------------
    '''
    sim.reset_meep()
    
    #Defining the final simulation object
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        symmetries=sym,
                        resolution=resolution,
                        Courant=courant)
    
    # reflected flux
    refl_t = sim.add_flux(fcen, df, nfreq, refl_fr_t)
    refl_b = sim.add_flux(fcen, df, nfreq, refl_fr_b)
    refl_l = sim.add_flux(fcen, df, nfreq, refl_fr_l)
    refl_r = sim.add_flux(fcen, df, nfreq, refl_fr_r)
    
    refl_up = sim.add_flux(fcen, df, nfreq, refl_fr_up)
    refl_dw = sim.add_flux(fcen, df, nfreq, refl_fr_dw)
    
    sim.use_output_directory('flux-sph_2')
    sim.run(mp.in_volume(monitor, mp.at_beginning(mp.output_epsilon)),
            mp.in_volume(monitor, mp.to_appended("ez", mp.at_every(time_step, mp.output_efield_x))),
            until_after_sources=mp.stop_when_fields_decayed(add_time,axis,pt,decay))
    
    #abs_refl_flux = mp.get_fluxes(refl)                                            
    #trans_flux_structure_abs.dat
    abs_refl_data_t = mp.get_fluxes(refl_t)
    abs_refl_data_b = mp.get_fluxes(refl_b)
    abs_refl_data_l = mp.get_fluxes(refl_l)
    abs_refl_data_r = mp.get_fluxes(refl_r)
    
    abs_refl_data_up = mp.get_fluxes(refl_up)
    abs_refl_data_dw = mp.get_fluxes(refl_dw)

    '''
    ------------------------------------------------
    Plotting the extintion, scattering and absorbtion
    ------------------------------------------------
    '''
    wl = []
    scat = []
    absoH = []
    absoV = []
    
    
    orig_flux=incident_tran_flux #make a copy of the data to avoid simulate again
    
    for k in range(len(orig_flux)):
        orig_flux[k] = abs(orig_flux[k])/ (4*rad) # 4*rad is the side of the flux monitor
    
    for i in range(0, nfreq):
        wl = np.append(wl, 1/flux_freqs[i]) # constructs the x axis wavelength
    
        scat_refl_flux = abs(scat_refl_data_t[i] - scat_refl_data_b[i] - scat_refl_data_l[i] + scat_refl_data_r[i] - scat_refl_data_up[i] +scat_refl_data_dw[i])
        scat = np.append(scat, scat_refl_flux/orig_flux[i])
        
        abs_refl_flux = abs(abs_refl_data_t[i] - abs_refl_data_b[i]) 
        absoV = np.append(absoV, abs_refl_flux/orig_flux[i])
        
        abs_refl_flux2 = abs(abs_refl_data_l[i] - abs_refl_data_r[i])
        absoH = np.append(absoH, abs_refl_flux2/orig_flux[i])
    
        abs_refl_flux3 = abs(abs_refl_data_up[i] - abs_refl_data_dw[i])
        absoZ = np.append(absoZ, abs_refl_flux3/orig_flux[i])
    
    plt.figure
    ext=scat + absoH + absoV + absoZ
    peaks, _ = find_peaks(ext, height=0.05)
    
    plt.plot(wl[peaks], ext[peaks], "x")
    plt.plot(wl,scat,'-',label='scatering')
    plt.plot(wl,absoH,'-',label='absorptionH')
    plt.plot(wl,absoV,'-',label='absorptionV')
    plt.plot(wl,absoZ,'-',label='absorptionZ')

    plt.plot(wl,ext, '-', label='extinction')
    
    #plt.plot(wl[peaksW], ext[peaks], "x")
    #plt.plot(wl,extW, '-', label='extinction with Water')
    
    
    plt.xlabel("wavelength (um)")
    plt.legend(loc="upper right")
    plt.show()
    

if runImages:
    
    '''
    --------------------------------
    Simulation with nanoparticle to get the images
    --------------------------------
    '''
    if runData:    
        sim.reset_meep()
    
    #Defining the final simulation object
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        symmetries=sym,
                        resolution=resolutionImages,
                        Courant=courant)
    #meep.Volume(center=meep.Vector3(0,0,0), size=meep.Vector3(10,0,0)), meep.output_dpwr
    sim.use_output_directory('nanosphere-imgs')
    sim.run(mp.in_volume(monitor2D, mp.at_beginning(mp.output_epsilon)),
            mp.in_volume(monitor2D, mp.at_every(time_step, mp.output_png(mp.Hz, "-Zc dkbluered"))),
            until_after_sources=mp.stop_when_fields_decayed(add_time,axis,pt,decay))
    
    print("Completed")