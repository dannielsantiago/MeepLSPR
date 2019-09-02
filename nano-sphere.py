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
import math
import h5py
import _shapes as shapes



'''
------------------------Parameters of the simulation
'''
rad=shapes.rad
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
monitorxy = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(mx,mx,0))
monitorxz = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(mx,0,mx))
f_response=True


#Inside the geometry object, the device structure is specified together with its center and the type of the material used
geometry = [shapes.sph]

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
refl_fr_t = mp.FluxRegion(center=mp.Vector3(0,fx/2,0), size=mp.Vector3(fx,0,fx)) # usado para normalizar tambien
refl_fr_b = mp.FluxRegion(center=mp.Vector3(0,-fx/2,0), size=mp.Vector3(fx,0,fx))
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
                    sources=sources,
                    resolution=resolution,
                    Courant=courant)

refl_t = sim.add_flux(fcen, df, nfreq, refl_fr_t)
refl_b = sim.add_flux(fcen, df, nfreq, refl_fr_b)
refl_l = sim.add_flux(fcen, df, nfreq, refl_fr_l)
refl_r = sim.add_flux(fcen, df, nfreq, refl_fr_r)
refl_up = sim.add_flux(fcen, df, nfreq, refl_fr_up)
refl_dw = sim.add_flux(fcen, df, nfreq, refl_fr_dw)
  
flux_freqs = np.array(mp.get_flux_freqs(refl_b))

if(f_response):
    #setup of the object in which the freq response will be stored via sim.add_dft_fields for all frequencies used in simulation
    dft_objxy=[]
    dft_objxz=[]
    for i in range(nfreq):
        dft_objxy=np.append(dft_objxy,sim.add_dft_fields([polarisation], flux_freqs[i], flux_freqs[i], 1, where=monitorxy))
        dft_objxz=np.append(dft_objxz,sim.add_dft_fields([polarisation], flux_freqs[i], flux_freqs[i], 1, where=monitorxz))
    
    
sim.use_output_directory('flux-sph')
sim.run(until_after_sources=mp.stop_when_fields_decayed(add_time,polarisation,pt,decay))

# for normalization run, save flux fields data for reflection plane
straight_refl_data_t = sim.get_flux_data(refl_t)
straight_refl_data_b = sim.get_flux_data(refl_b)
straight_refl_data_l = sim.get_flux_data(refl_l)
straight_refl_data_r = sim.get_flux_data(refl_r)
straight_refl_data_up = sim.get_flux_data(refl_up)
straight_refl_data_dw = sim.get_flux_data(refl_dw)

# save incident flux for transmission plane
incident_flux_b = np.array(mp.get_fluxes(refl_b))
incident_flux_t = np.array(mp.get_fluxes(refl_t)) 
incident_flux_l = np.array(mp.get_fluxes(refl_l))
incident_flur_r = np.array(mp.get_fluxes(refl_r))
incident_flur_up = np.array(mp.get_fluxes(refl_up))
incident_flur_dw = np.array(mp.get_fluxes(refl_dw))

if(f_response):
    #get real and imaginary parts of all frequency responses for each frequency 
    ex0_xy_r = []
    ex0_xy_i = []
    ex0_xz_r = []
    ex0_xz_i = []
    
    for i in range(nfreq):
        ex0_xy_r=np.append(ex0_xy_r,np.real(sim.get_dft_array(dft_objxy[i], polarisation, 0))) 
        ex0_xy_i=np.append(ex0_xy_i,np.imag(sim.get_dft_array(dft_objxy[i], polarisation, 0)))
        ex0_xz_r=np.append(ex0_xz_r,np.real(sim.get_dft_array(dft_objxz[i], polarisation, 0))) 
        ex0_xz_i=np.append(ex0_xz_i,np.imag(sim.get_dft_array(dft_objxz[i], polarisation, 0)))

    size=int(np.sqrt(len(ex0_xy_r)/nfreq))
    ex0_xy_r=ex0_xy_r.reshape((nfreq,size,size))
    ex0_xy_i=ex0_xy_i.reshape((nfreq,size,size))
    ex0_xz_r=ex0_xz_r.reshape((nfreq,size,size))
    ex0_xz_i=ex0_xz_i.reshape((nfreq,size,size))
    
 
    ex0_xy_complex=np.zeros_like(ex0_xy_r, dtype=complex)
    ex0_xz_complex=np.zeros_like(ex0_xz_r, dtype=complex)

    ex0_xy_complex.real=ex0_xy_r
    ex0_xy_complex.imag=ex0_xy_i
    del ex0_xy_r
    del ex0_xy_i
    ex0_xz_complex.real=ex0_xz_r
    ex0_xz_complex.imag=ex0_xz_i
    del ex0_xz_r
    del ex0_xz_i
    
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

if(f_response):
    #setup of the object in which the freq response will be stored via sim.add_dft_fields for all frequencies used in simulation
    dft_objxy=[]
    dft_objxz=[]
    for i in range(nfreq):
        dft_objxy=np.append(dft_objxy,sim.add_dft_fields([polarisation], flux_freqs[i], flux_freqs[i], 1, where=monitorxy))
        dft_objxz=np.append(dft_objxz,sim.add_dft_fields([polarisation], flux_freqs[i], flux_freqs[i], 1, where=monitorxz))
        
sim.use_output_directory('flux-sph')
sim.run(until_after_sources=mp.stop_when_fields_decayed(add_time,polarisation,pt,decay))
        #until=until)

#get reflected flux from the surfaces
scat_refl_data_t = np.array(mp.get_fluxes(refl_t))
scat_refl_data_b = np.array(mp.get_fluxes(refl_b))
scat_refl_data_l = np.array(mp.get_fluxes(refl_l))
scat_refl_data_r = np.array(mp.get_fluxes(refl_r))
scat_refl_data_up = np.array(mp.get_fluxes(refl_up))
scat_refl_data_dw = np.array(mp.get_fluxes(refl_dw))

#get absorbed fluxes from the surfaces
abso_refl_data_t = np.array(mp.get_fluxes(arefl_t))
abso_refl_data_b = np.array(mp.get_fluxes(arefl_b))
abso_refl_data_l = np.array(mp.get_fluxes(arefl_l))
abso_refl_data_r = np.array(mp.get_fluxes(arefl_r))
abso_refl_data_up = np.array(mp.get_fluxes(arefl_up))
abso_refl_data_dw = np.array(mp.get_fluxes(arefl_dw))

if(f_response):
    #get real and imaginary parts of all frequency responses for each frequency 
    ex_xy_r = []
    ex_xy_i = []
    ex_xz_r = []
    ex_xz_i = []
    
    for i in range(nfreq):
        ex_xy_r=np.append(ex_xy_r,np.real(sim.get_dft_array(dft_objxy[i], polarisation, 0))) 
        ex_xy_i=np.append(ex_xy_i,np.imag(sim.get_dft_array(dft_objxy[i], polarisation, 0)))
        ex_xz_r=np.append(ex_xz_r,np.real(sim.get_dft_array(dft_objxz[i], polarisation, 0))) 
        ex_xz_i=np.append(ex_xz_i,np.imag(sim.get_dft_array(dft_objxz[i], polarisation, 0)))

    ex_xy_r=ex_xy_r.reshape((nfreq,size,size))
    ex_xy_i=ex_xy_i.reshape((nfreq,size,size))
    ex_xz_r=ex_xz_r.reshape((nfreq,size,size))
    ex_xz_i=ex_xz_i.reshape((nfreq,size,size))
    
    ex_xy_complex=np.zeros_like(ex_xy_r, dtype=complex)
    ex_xz_complex=np.zeros_like(ex_xz_r, dtype=complex)

    ex_xy_complex.real=ex_xy_r
    ex_xy_complex.imag=ex_xy_i
    del ex_xy_r
    del ex_xy_i
    ex_xz_complex.real=ex_xz_r
    ex_xz_complex.imag=ex_xz_i
    del ex_xz_r
    del ex_xz_i

'''
------------------------------------------------
Plotting the extintion, scattering and absorbtion
------------------------------------------------
'''
wl = 1/flux_freqs
flux_side=fx*1000
flux_area=flux_side**2
sph_area=math.pi*(rad*1000)**2

incidentPow=incident_flux_b/flux_area
#cross_sections
scat=(abs(scat_refl_data_t - scat_refl_data_b) + abs(scat_refl_data_l - scat_refl_data_r) + abs(scat_refl_data_up - scat_refl_data_dw))/incidentPow
abso=(abs(abso_refl_data_t - abso_refl_data_b) + abs(abso_refl_data_l - abso_refl_data_r) + abs(abso_refl_data_up - abso_refl_data_dw))/incidentPow
ext=scat + abso
#efficiencies
scat/=sph_area
abso/=sph_area
ext/=sph_area


norm=incidentPow/incidentPow.max()
    
#saving simulation data in external file
Fd_file = h5py.File('flux-sph-600/cross-sections-25nm.h5','w')
Fd_file.create_dataset('wl',data=wl)
Fd_file.create_dataset('ext',data=ext)
Fd_file.create_dataset('scat',data=scat)
Fd_file.create_dataset('abso',data=abso)
Fd_file.create_dataset('rad',data=rad)
Fd_file.create_dataset('resolution',data=resolution)
Fd_file.close()

Fd_file = h5py.File('flux-sph-600/scat-fx-25nm.h5','w')
Fd_file.create_dataset('scat_up',data=scat_refl_data_up)
Fd_file.create_dataset('scat_dw',data=scat_refl_data_dw)
Fd_file.create_dataset('scat_b',data=scat_refl_data_b)
Fd_file.create_dataset('scat_t',data=scat_refl_data_t)
Fd_file.create_dataset('scat_l',data=scat_refl_data_l)
Fd_file.create_dataset('scat_r',data=scat_refl_data_r)
Fd_file.close()

plt.figure()
plt.plot(wl,scat,'ob',label='scatering')
plt.plot(wl,abso,'sr',label='absorption')
plt.plot(wl,ext,'^g', label='extinction')

#Analytical model
x=c_coreshell(wl*1000,'Ag','Ag',1,20,5,5)

plt.plot(x[0:,0],x[0:,1], '-k',label='Analytical model')
plt.plot(x[0:,0],x[0:,2], '-k')
plt.plot(x[0:,0],x[0:,3], '-k')

radx=rad*1000
plt.title('Efficiencies of Silver sphere of radius %inm' %radx)
plt.xlabel("wavelength (um)")
plt.ylabel("Efficiencies")
plt.legend(loc="upper right")  
plt.axis([0.31, 0.7, 0, max(ext)*1.2])
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

if(f_response):
    wl_r=wl[::-1]
    #FT with particle
    #function to plot an array vs the eps_data(wire) 
    def showMultiple(array,title,row=5,col=5,cmap='jet',log=False,domain='frequency'):
        row_=row
        col_=col
        N=row_*col_
        x=np.linspace(0,99,N)
        print(x)
        plt.figure()
        plt.suptitle(title)
        for i in range(N):
            plt.subplot(row,col,i+1)
            #plt.imshow(eps_data.T, interpolation='spline36', cmap='binary')
            plt.imshow(array[int(x[i])].T, interpolation='spline36', cmap=cmap, alpha=0.9, vmin=array.min(), vmax=10)
            if domain=='frequency':    
                plt.ylabel(str(int(wl_r[int(x[i])]*1000))+' nm')
            elif domain=='time':
                plt.ylabel('t= '+str(int(i*len(array)/N)))
            #plt.colorbar()
            #plt.axis('off')
        plt.show()
        
    def showMaxSlice(array,title,cmap='jet'):
        #shows the slice with the maximum value
        xslice,xpos,ypos=np.unravel_index(array.argmax(), array.shape)
        print(xslice)
        plt.figure()
        plt.suptitle(title)
        plt.imshow(array[xslice].T, interpolation='spline36', cmap=cmap, vmin=array.min(), vmax=10)
        plt.colorbar()
        plt.show()
        
    #save the frequency response in a file
    Fd_file = h5py.File('flux-sph-600/F_response-xy-25nm.h5','w')
    Fd_file.create_dataset('FT_Ex_r',data=ex_xy_complex.real.T)
    Fd_file.create_dataset('FT_Ex_i',data=ex_xy_complex.imag.T)
    Fd_file.create_dataset('FT_Ex0_r',data=ex0_xy_complex.real.T)
    Fd_file.create_dataset('FT_Ex0_i',data=ex0_xy_complex.imag.T)
    Fd_file.close()
    Fd_file = h5py.File('flux-sph-600/F_response-xz-25nm.h5','w')
    Fd_file.create_dataset('FT_Ex_r',data=ex_xz_complex.real.T)
    Fd_file.create_dataset('FT_Ex_i',data=ex_xz_complex.imag.T)
    Fd_file.create_dataset('FT_Ex0_r',data=ex0_xz_complex.real.T)
    Fd_file.create_dataset('FT_Ex0_i',data=ex0_xz_complex.imag.T)
    Fd_file.close()
    
    data_diff=np.real(ex_xy_complex*np.conj(ex_xy_complex))
    data_diff2=np.real(ex0_xy_complex*np.conj(ex0_xy_complex))
    data_diff3=data_diff/data_diff2
    showMultiple(data_diff3,title='E/E0 - xy')
    #showMultiple(data_diff3,row=3, col=3,title='E/E0 - xy')
    #showMaxSlice(data_diff3,title='E/E0 - xy')

    data_diff=np.real(ex_xz_complex*np.conj(ex_xz_complex))
    data_diff2=np.real(ex0_xz_complex*np.conj(ex0_xz_complex))
    data_diff3=data_diff/data_diff2
    showMultiple(data_diff3,title='E/E0 - xz')
    #showMultiple(data_diff3,row=3, col=3,title='E/E0 - xy')
    #showMaxSlice(data_diff3,title='E/E0 - xy')

