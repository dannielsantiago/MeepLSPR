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
#import _shapes as shapes
from coreShell import c_coreshell
from materials_library import Ag, Au, cSi, myAg

import h5py



'''
------------------------Parameters of the simulation
'''
rad=0.025
w=0.43                  # wavelength
fcen=1/w                # Pulse center frequency
df = 1.8                  # pulse width in micrometers

polarisation=mp.Ex              # Axis of direction of the pulse Ex=TM, Hx=TE
dpml = 0.7015/2                # Width of th pml layers = wavelength

mx = 0.2             # size of monitor box side 8*rad
fx = 4*rad              # size of flux box side

sx = mx+0.01             # Size of inner shell
sy = mx+0.01             # Size of inner shell
sz = mx+0.01             # Size of inner shell

sx0 = sx + 2*dpml       # size of cell in X direction
sy0 = sy + 2*dpml       # size of cell in Y direction
sz0 = sz + 2*dpml       # size of cell in z direction

sc=mx/2 + 0.005           # source center coordinate shift
sw=sx0                  # source width, needs to be bigger than inner cell to generate only plane-waves

nfreq = 100             # number of frequencies at which to compute flux
courant=0.5            # numerical stability, default is 0.5, should be lower in case refractive index n<1

time_step=0.1           # time step to measure flux
add_time=2             # additional time until field decays 1e-6
resolution = 600        # resolution pixels/um (pixels/micrometers)
decay = 1e-12           # decay limit condition for the field measurement

cell = mp.Vector3(sx0, sy0, sz0) 
monitor = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(mx,mx,0))


#Inside the geometry object, the device structure is specified together with its center and the type of the material used
def sph_resonator(p):
    rr = (p.x**2+p.y**2)**0.5
    if (rr < rad):
        return Ag
    return mp.air

sph_resonator.do_averaging = True

geometry = [mp.Block(center=mp.Vector3(),
                     size=mp.Vector3(sx0,sx0,sx0),
                     material=sph_resonator)]
#geometry = [shapes.sph]

#Boundary conditions using Perfectly Maching Layers PML// ficticius absorbtion material to avoid reflection of the fields
pml_layers = [mp.PML(dpml)]

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
                     component=polarisation,
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



'''
------------------------------
1st simulaton without particle and Get normal flux
------------------------------
'''

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=[],
                    subpixel_tol=1e-4,
                    subpixel_maxeval=1000,
                    sources=sources,
                    resolution=resolution,
                    Courant=courant)

refl_t = sim.add_flux(fcen, df, nfreq, refl_fr_t)
refl_b = sim.add_flux(fcen, df, nfreq, refl_fr_b)
refl_l = sim.add_flux(fcen, df, nfreq, refl_fr_l)
refl_r = sim.add_flux(fcen, df, nfreq, refl_fr_r)

refl_up = sim.add_flux(fcen, df, nfreq, refl_fr_up)
refl_dw = sim.add_flux(fcen, df, nfreq, refl_fr_dw)
    
sim.use_output_directory('flux-sph600')
sim.run(mp.in_volume(monitor, mp.at_beginning(mp.output_epsilon)),
        until_after_sources=mp.stop_when_fields_decayed(add_time,polarisation,pt,decay))

# for normalization run, save flux fields data for reflection plane
straight_refl_data_t = sim.get_flux_data(refl_t)
straight_refl_data_b = sim.get_flux_data(refl_b)
straight_refl_data_l = sim.get_flux_data(refl_l)
straight_refl_data_r = sim.get_flux_data(refl_r)

straight_refl_data_up = sim.get_flux_data(refl_up)
straight_refl_data_dw = sim.get_flux_data(refl_dw)

# save incident flux for transmission plane
incident_tran_flux = mp.get_fluxes(refl_b)

'''
------------------------------------------------
2nd Simulation with particle and get Scattered Flux
------------------------------------------------
'''
sim.reset_meep()

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    subpixel_tol=1e-4,
                    subpixel_maxeval=1000,
                    sources=sources,
                    resolution=resolution,
                    Courant=courant)

# reflected flux
refl_t = sim.add_flux(fcen, df, nfreq, refl_fr_t)
refl_b = sim.add_flux(fcen, df, nfreq, refl_fr_b)
refl_l = sim.add_flux(fcen, df, nfreq, refl_fr_l)
refl_r = sim.add_flux(fcen, df, nfreq, refl_fr_r)

refl_up = sim.add_flux(fcen, df, nfreq, refl_fr_up)
refl_dw = sim.add_flux(fcen, df, nfreq, refl_fr_dw)

# absorbed flux
arefl_t = sim.add_flux(fcen, df, nfreq, refl_fr_t)
arefl_b = sim.add_flux(fcen, df, nfreq, refl_fr_b)
arefl_l = sim.add_flux(fcen, df, nfreq, refl_fr_l)
arefl_r = sim.add_flux(fcen, df, nfreq, refl_fr_r)

arefl_up = sim.add_flux(fcen, df, nfreq, refl_fr_up)
arefl_dw = sim.add_flux(fcen, df, nfreq, refl_fr_dw)

# for normal run, load negated fields to subtract incident from refl. fields
sim.load_minus_flux_data(refl_t, straight_refl_data_t)
sim.load_minus_flux_data(refl_b, straight_refl_data_b)
sim.load_minus_flux_data(refl_l, straight_refl_data_l)
sim.load_minus_flux_data(refl_r, straight_refl_data_r)

sim.load_minus_flux_data(refl_up, straight_refl_data_up)
sim.load_minus_flux_data(refl_dw, straight_refl_data_dw)


sim.use_output_directory('flux-sph600')
sim.run(mp.in_volume(monitor, mp.at_beginning(mp.output_epsilon)),
        until_after_sources=mp.stop_when_fields_decayed(add_time,polarisation,pt,decay))
        #until=until)

#get reflected flux from the surfaces
scat_refl_data_t = mp.get_fluxes(refl_t)
scat_refl_data_b = mp.get_fluxes(refl_b)
scat_refl_data_l = mp.get_fluxes(refl_l)
scat_refl_data_r = mp.get_fluxes(refl_r)

scat_refl_data_up = mp.get_fluxes(refl_up)
scat_refl_data_dw = mp.get_fluxes(refl_dw)

#get absorbed fluxes from the surfaces
abs_refl_data_t = mp.get_fluxes(arefl_t)
abs_refl_data_b = mp.get_fluxes(arefl_b)
abs_refl_data_l = mp.get_fluxes(arefl_l)
abs_refl_data_r = mp.get_fluxes(arefl_r)

abs_refl_data_up = mp.get_fluxes(arefl_up)
abs_refl_data_dw = mp.get_fluxes(arefl_dw)

flux_freqs = mp.get_flux_freqs(arefl_b)


'''
------------------------------------------------
Plotting the extintion, scattering and absorbtion
------------------------------------------------
'''
wl = []
scat = []
abso = []
ext=[]

for i in range(0, nfreq):
    wl = np.append(wl, 1/flux_freqs[i]) # constructs the x axis wavelength

    scat_refl_flux = abs(scat_refl_data_t[i] - scat_refl_data_b[i]) + abs(scat_refl_data_l[i] - scat_refl_data_r[i]) + abs(scat_refl_data_up[i] - scat_refl_data_dw[i])
    scat = np.append(scat, scat_refl_flux/incident_tran_flux[i])
    
    abs_refl_flux = abs(abs_refl_data_t[i] - abs_refl_data_b[i]) + abs(abs_refl_data_l[i] - abs_refl_data_r[i]) + abs(abs_refl_data_up[i] - abs_refl_data_dw[i])
    abso = np.append(abso, abs_refl_flux/incident_tran_flux[i])
    

#multily for area of the sides to get the crossection in nm,
# area=4*rad=100 nm
scat=scat*rad*rad*10000
abso=abso*rad*rad*10000
ext=scat + abso

#saving simulation data in external file
Fd_file = h5py.File('flux-sph/cross-sections-25nm600.h5','w')
Fd_file.create_dataset('wl',data=wl)
Fd_file.create_dataset('ext',data=ext)
Fd_file.create_dataset('scat',data=scat)
Fd_file.create_dataset('abso',data=abso)
Fd_file.create_dataset('rad',data=rad)
Fd_file.create_dataset('resolution',data=resolution)
Fd_file.close()


plt.figure()
plt.plot(wl,scat,'ob',label='scatering')
plt.plot(wl,abso,'sr',label='absorption')
plt.plot(wl,ext,'^g', label='extinction')

#Analytical model
x=c_coreshell(wl*1000,'Ag','Ag',1,15,10,5)

plt.plot(x[0:,0],x[0:,1], '-k',label='Analytical model')
plt.plot(x[0:,0],x[0:,2], '-k')
plt.plot(x[0:,0],x[0:,3], '-k')

radx=rad*1000
plt.title('Efficiencies of Silver sphere of radius %inm' %radx)
plt.xlabel("wavelength (um)")
plt.ylabel("Efficiencies")
plt.legend(loc="upper right")  
plt.axis([0.24, 0.7, 0, max(ext)*1.2])
plt.grid(True)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

'''
------------------------------
Section to plot fiting error 
------------------------------
'''        
y=np.array(ext)[np.newaxis].T
yref=x[0:,1]

deltaSignal = abs(y - yref)
error = (deltaSignal/yref)*100 # Percent by element. *100
plt.figure()  
plt.plot(wl,error, '--r', label='% error')
#plt.plot(wl,np.ones(len(wl))*meanPctDiff,'r', label='%.2f%% avg error' %meanPctDiff)
plt.axis([0.3, 0.7, 0, max(error)*1.2])  
plt.grid(True)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel("wavelength (um)")
plt.ylabel('% error', color='r')  
plt.tick_params('y', colors='r')
#plt.legend(loc="center right")  
plt.show()
print(np.mean(error))

print(error.max()) 

