#Import libraries
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import _shapes as shapes
from scipy.signal import find_peaks
import h5py

import c_wire90 as theoretical

'''
------------------------Parameters of the simulation
'''
rad=shapes.rad
w=0.357                  # wavelength
fcen=1/w                # Pulse center frequency
df = 3.5                # 3,5pulse frequency width 
cutoff=10
polarisation=mp.Ex      # Axis of direction of the pulse Ex=TM, Hx=TE
dpml = w                # Width of th pml layers = wavelength
sx = 14*rad             # Size of inner shell
sy = 14*rad             # Size of inner shell
sx0 = sx + 2*dpml       # size of cell in X direction
sy0 = sy + 2*dpml       # size of cell in Y direction
mx = 8*rad
fx = 4*rad
sc=6*rad                # source center coordinate shift
sw=sx0                  # source width, needs to be bigger than inner cell to generate only plane-waves
nfreq = 200             # number of frequencies at which to compute flux
courant=0.5            # numerical stability, default is 0.5, should be lower in case refractive index n<1
time_step=0.05           # time step to measure flux
add_time=2             # additional time until field decays 1e-6
resolution =250        # resolution pixels/um (pixels/micrometers)
decay = 1e-12           # decay limit condition for the field measurement
cell = mp.Vector3(sx0, sy0, 0) 
monitor = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(mx,mx,0))
until=23                #should be at least 22 time units
'''

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


#Inside the geometry object, the device structure is specified together with its center and the type of the material used
geometry = [shapes.cyl]

#Boundary conditions using Perfectly Maching Layers PML// ficticius absorbtion material to avoid reflection of the fields
pml_layers = [mp.PML(dpml)]

'''
y              
^                
|                
|           
|              
+-------> x  
Gaussian       
'''

# Defining the sources, the y-axis is inverted so propagation direction is down
gaussian=mp.Source(mp.GaussianSource(wavelength=w, fwidth=df, cutoff=cutoff),
                     component=polarisation,
                     center=mp.Vector3(0,-sc,0),
                     amplitude=1,
                     size=mp.Vector3(sw,0,0))

pt=mp.Vector3(0,sc,0) # 1.1*radpoint used to measure decay of Field (in oposite side of the source)



'''
-------------------------------
Regions for the flux measurement 
-------------------------------

                    source
               +---------------+
                    
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
            
'''
# reflected flux / Regions (top,bottom,left,right)
refl_fr_t = mp.FluxRegion(center=mp.Vector3(0,-fx/2,0), size=mp.Vector3(fx,0,0)) # usado para normalizar tambien
refl_fr_b = mp.FluxRegion(center=mp.Vector3(0,fx/2,0), size=mp.Vector3(fx,0,0))
refl_fr_l = mp.FluxRegion(center=mp.Vector3(-fx/2,0,0), size=mp.Vector3(0,fx,0))
refl_fr_r = mp.FluxRegion(center=mp.Vector3(fx/2,0,0), size=mp.Vector3(0,fx,0))

sources = [gaussian]

'''
------------------------------
1st simulaton without particle and Get normal flux
------------------------------
'''

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=[],
                    sources=sources,
                    force_complex_fields=True,
                    resolution=resolution,
                    Courant=courant)


refl_t = sim.add_flux(fcen, df, nfreq, refl_fr_t) 
refl_b = sim.add_flux(fcen, df, nfreq, refl_fr_b)
refl_l = sim.add_flux(fcen, df, nfreq, refl_fr_l)
refl_r = sim.add_flux(fcen, df, nfreq, refl_fr_r)

#get array of all frequencies used in simulation
flux_freqs = np.array(mp.get_flux_freqs(refl_b))

#setup of the object in which the freq response will be stored via sim.add_dft_fields for all frequencies used in simulation
dft_objx=[]
for i in range(nfreq):
    dft_objx=np.append(dft_objx,sim.add_dft_fields([polarisation], flux_freqs[i], flux_freqs[i], 1, where=monitor))
    
#sim.run(until_after_sources=mp.stop_when_fields_decayed(add_time,polarisation,pt,decay))
sim.use_output_directory('flux-out')
sim.run(mp.in_volume(monitor, mp.at_beginning(mp.output_epsilon)),
        mp.in_volume(monitor, mp.to_appended("ex0", mp.at_every(time_step, mp.output_efield_x))),
        until=until)
# for normalization run, save flux fields data for reflection plane
straight_refl_data_t = sim.get_flux_data(refl_t)
straight_refl_data_b = sim.get_flux_data(refl_b)
straight_refl_data_l = sim.get_flux_data(refl_l)
straight_refl_data_r = sim.get_flux_data(refl_r)

#get initial flux without particle to compute the ratio against the scattered fluxes with particle
incident_flux = mp.get_fluxes(refl_b) 

#get real and imaginary parts of all frequency responses for each frequency 
ex0_arr_real = []
for i in range(nfreq):
    ex0_arr_real=np.append(ex0_arr_real,np.real(sim.get_dft_array(dft_objx[i], polarisation, 0)))   
size=int(np.sqrt(len(ex0_arr_real)/nfreq))
ex0_arr_real=ex0_arr_real.reshape((nfreq,size,size))

ex0_arr_imag = []
for i in range(len(flux_freqs)):
    ex0_arr_imag=np.append(ex0_arr_imag,np.imag(sim.get_dft_array(dft_objx[i], polarisation, 0)))
ex0_arr_imag=ex0_arr_imag.reshape((nfreq,size,size))

ex0_arr_complex=np.zeros_like(ex0_arr_real, dtype=complex)
ex0_arr_complex.real=ex0_arr_real
ex0_arr_complex.imag=ex0_arr_imag
#ex0_arr_complex=ex0_arr_imag*np.conj(ex0_arr_imag)

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
                    force_complex_fields=True,
                    resolution=resolution,
                    Courant=courant)

# scattered flux
refl_t = sim.add_flux(fcen, df, nfreq, refl_fr_t)
refl_b = sim.add_flux(fcen, df, nfreq, refl_fr_b)
refl_l = sim.add_flux(fcen, df, nfreq, refl_fr_l)
refl_r = sim.add_flux(fcen, df, nfreq, refl_fr_r)

# absorbance flux
arefl_t = sim.add_flux(fcen, df, nfreq, refl_fr_t)
arefl_b = sim.add_flux(fcen, df, nfreq, refl_fr_b)
arefl_l = sim.add_flux(fcen, df, nfreq, refl_fr_l)
arefl_r = sim.add_flux(fcen, df, nfreq, refl_fr_r)

# for normal run, load negated fields to subtract incident from refl. fields
sim.load_minus_flux_data(refl_t, straight_refl_data_t)
sim.load_minus_flux_data(refl_b, straight_refl_data_b)
sim.load_minus_flux_data(refl_l, straight_refl_data_l)
sim.load_minus_flux_data(refl_r, straight_refl_data_r)

#setup of the object in which the freq response will be stored via sim.add_dft_fields for all frequencies used in simulation
dft_objx=[]
for i in range(len(flux_freqs)):
    dft_objx=np.append(dft_objx,sim.add_dft_fields([polarisation], flux_freqs[i], flux_freqs[i], 1, where=monitor))
    
sim.use_output_directory('flux-out')
sim.run(mp.in_volume(monitor, mp.at_beginning(mp.output_epsilon)),
        mp.in_volume(monitor, mp.to_appended("ex", mp.at_every(time_step, mp.output_efield_x))),
        #until_after_sources=mp.stop_when_fields_decayed(add_time,polarisation,pt,decay))
        until=until)
'''
sim.run(until_after_sources=mp.stop_when_fields_decayed(add_time,polarisation,pt,decay))
'''

#save scattered reflected flux from the surfaces
scat_refl_data_t = mp.get_fluxes(refl_t)
scat_refl_data_b = mp.get_fluxes(refl_b)
scat_refl_data_l = mp.get_fluxes(refl_l)
scat_refl_data_r = mp.get_fluxes(refl_r)

#trans_flux_structure_abs.dat
abs_refl_data_t = mp.get_fluxes(arefl_t)
abs_refl_data_b = mp.get_fluxes(arefl_b)
abs_refl_data_l = mp.get_fluxes(arefl_l)
abs_refl_data_r = mp.get_fluxes(arefl_r)

# save incident power for transmission plane

transmitted_flux = abs_refl_data_b

eps_data = sim.get_array(vol=monitor, component=mp.Dielectric)

ex_arr_real = []
for i in range(len(flux_freqs)):
    ex_arr_real=np.append(ex_arr_real,np.real(sim.get_dft_array(dft_objx[i], polarisation, 0)))
ex_arr_real=ex_arr_real.reshape((nfreq,size,size))

ex_arr_imag = []
for i in range(len(flux_freqs)):
    ex_arr_imag=np.append(ex_arr_imag,np.imag(sim.get_dft_array(dft_objx[i], polarisation, 0)))
ex_arr_imag=ex_arr_imag.reshape((nfreq,size,size))

ex_arr_complex=np.zeros_like(ex_arr_real, dtype=complex)
ex_arr_complex.real=ex_arr_real
ex_arr_complex.imag=ex_arr_imag
#ex_arr_complex=ex_arr_imag*np.conj(ex_arr_imag)
'''
------------------------------------------------
Plotting the extintion, scattering and absorbtion
------------------------------------------------
'''
wl = []
scat = []
abso = []
norm = []
tran=[]
scatt=[]
ext=[]
mat=[]


for i in range(0, nfreq):
    wl = np.append(wl, 1/flux_freqs[i]) # constructs the x axis wavelength

    scat_refl_flux = abs(scat_refl_data_t[i] - scat_refl_data_b[i] + scat_refl_data_l[i]- scat_refl_data_r[i])
    scat = np.append(scat, scat_refl_flux/incident_flux[i])

    abs_refl_flux = abs(abs_refl_data_t[i] - abs_refl_data_b[i] + abs_refl_data_l[i] - abs_refl_data_r[i])
    abso = np.append(abso, abs_refl_flux/incident_flux[i])
    
    norm = np.append(norm, incident_flux[i]/max(incident_flux))
    tran = np.append(tran, transmitted_flux[i]/max(incident_flux))
    scatt = np.append(scatt, scat_refl_data_b[i]/max(incident_flux))

#multily for area (lenght in this case) of the sides to get the crossection in nm,
# area=4*rad=100 nm
scat=scat*4*rad*1000
abso=abso*4*rad*1000
ext=scat + abso

#saving simulation data in external file
Fd_file = h5py.File('flux-out/cross-sections-25nm.h5','w')
Fd_file.create_dataset('wl',data=wl)
Fd_file.create_dataset('ext',data=ext)
Fd_file.create_dataset('scat',data=scat)
Fd_file.create_dataset('abso',data=abso)
Fd_file.create_dataset('rad',data=rad)
Fd_file.create_dataset('resolution',data=resolution)
Fd_file.close()

'''
---------------------------------
#calculate theoretical behaviour
check c_wire90.py for instructions
---------------------------------
'''
mat=theoretical.C_wire90(wl*1000,'Ag',1,shapes.rad*1000,16,'TM')
'''
------------------------------------------------------
Section to plot Meep simulation and Analytical response 
------------------------------------------------------ 
''' 
plt.figure()
plt.plot(wl,scat,'ob',label='scatering')
plt.plot(wl,abso,'sr',label='absorption')
plt.plot(wl,ext,'^g', label='extinction')

plt.plot(mat[0:,0],mat[0:,1], '-k', label='Analytical model')
plt.plot(mat[0:,0],mat[0:,2], '-k')
plt.plot(mat[0:,0],mat[0:,3], '-k')
'''
plt.plot(wl,norm, '-', label='incident pulse', linestyle='--')
plt.plot(wl,tran, '-', label='transmitted pulse', linestyle='--')
plt.plot(wl,scatt, '-', label='scatt pulse', linestyle='--')
'''
radx=rad*1000
plt.title('Cross-sections of Silver Nanowire of radius %inm and TM polarisation' %radx)
plt.xlabel("wavelength (um)")
plt.ylabel("cross-section (nm)")
plt.legend(loc="upper right")  
plt.axis([0.3, 0.7, 0, max(mat[0:,1])*1.2])
plt.grid(True)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
   
'''
------------------------------
Section to plot fiting error 
------------------------------
'''        
y=np.array(ext)[np.newaxis].T
yref=mat[0:,1]

deltaSignal = abs(y - yref)
percentageDifference = (deltaSignal/yref)*100 # Percent by element. *100
plt.figure()  
plt.plot(wl,percentageDifference, '--r', label='% error')
#plt.plot(wl,np.ones(len(wl))*meanPctDiff,'r', label='%.2f%% avg error' %meanPctDiff)
plt.axis([0.3, 0.7, 0, max(percentageDifference)*1.2])  
plt.grid(True)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel("wavelength (um)")
plt.ylabel('% error', color='r')  
plt.tick_params('y', colors='r')
#plt.legend(loc="center right")  
plt.show()



'''
-----------------------------------
Section to plot frequency response
---------------------------------- 
'''
#FT with particle
#function to plot an array vs the eps_data(wire) 
def showMultiple(array,title,row=5,col=5,cmap='jet',log=False,domain='frequency'):
    row_=row
    col_=col
    N=row_*col_
    x=np.linspace(len(array)-1,0,N)
    print(x)
    plt.figure()
    plt.suptitle(title)
    for i in range(N):
        plt.subplot(row,col,i+1)
        plt.imshow(eps_data.T, interpolation='spline36', cmap='binary')
        if log:
            plt.imshow(array[int(x[i])].T, interpolation='spline36', cmap=cmap, alpha=0.9,  norm=colors.LogNorm(vmin=array.min(), vmax=array.max()))
        else:
            plt.imshow(array[int(x[i])].T, interpolation='spline36', cmap=cmap, alpha=0.9, vmin=array.min(), vmax=array.max())
        if domain=='frequency':    
            plt.ylabel(str(int(1000/flux_freqs[int(x[i])]))+' nm')
        elif domain=='time':
            plt.ylabel('t= '+str(int(i*len(array)/N)))
        plt.colorbar()
    plt.show()
    

#save the frequency response in a file
Fd_file = h5py.File('flux-out/F_response.h5','w')
Fd_file.create_dataset('FT_Ex_r',data=ex_arr_real.T)
Fd_file.create_dataset('FT_Ex_i',data=ex_arr_imag.T)

Fd_file.create_dataset('FT_Ex0_r',data=ex0_arr_real.T)
Fd_file.create_dataset('FT_Ex0_i',data=ex0_arr_imag.T)

Fd_file.close()

data_diff=np.real((ex_arr_complex-ex0_arr_complex)*np.conj(ex_arr_complex-ex0_arr_complex))
data_diff2=np.real(ex0_arr_complex*np.conj(ex0_arr_complex))
data_diff3=data_diff/data_diff2
data_diff4=np.log10(np.square(data_diff3))

    
#showMultiple(data_diff,title='Ex_c*cc')
#showMultiple(data_diff2,title='Ex0_c*cc')
#showMultiple(data_diff3,title='Ex_c*cc/Ex0_c*cc')
showMultiple(data_diff4,title='log10((Ex_c*cc/Ex0_c*cc)**2)')


def showMaxSlice(array,title,cmap='jet'):
    #shows the slice with the maximum value
    xslice,xpos,ypos=np.unravel_index(array.argmax(), array.shape)
    plt.figure()
    plt.suptitle(title)
    plt.imshow(array[xslice].T, interpolation='spline36', cmap=cmap, vmin=array.min(), vmax=array.max())
    plt.colorbar()
    plt.show()

showMaxSlice(data_diff,title='Ex_c*cc')
showMaxSlice(data_diff2,title='Ex0_c*cc')
showMaxSlice(data_diff3,title='Ex_c*cc/Ex0_c*cc')
showMaxSlice(data_diff4,title='log10((Ex_c*cc/Ex0_c*cc)**2)')


'''
-------------------------------------
Section to plot time-stepped response 
--------------------------------------
'''

#load Electric fields generated by the simulation and compute the difference
E0_file = h5py.File("flux-out/nanowire-ex0.h5")
E_file = h5py.File("flux-out/nanowire-ex.h5")
print(list(E0_file))
print(list(E_file))

E0_r=np.array(E0_file.get('ex.r')).T
E0_i=np.array(E0_file.get('ex.i')).T
E0_c=np.zeros_like(E0_r, dtype=complex)
E0_c.real=E0_r
E0_c.imag=E0_i
    
E_r=np.array(E_file.get('ex.r')).T
E_i=np.array(E_file.get('ex.i')).T
E_c=np.zeros_like(E_r, dtype=complex)
E_c.real=E_r
E_c.imag=E_i

#They must be the same size
print(len(E0_r[1]))
print(np.shape(E0_r[1]))
E0_file.close()
E_file.close()

#write in .h5 file the difference of electrics fields than later can be converted to images using terminal commands

Ed=E_r-E0_r
Ed_file = h5py.File('flux-out/E_diff.h5','w')
Ed_file.create_dataset('ex',data=Ed.T)
Ed_file.close()
showMaxSlice(Ed,title='time response',cmap='RdBu')
Ed=E_c*np.conj(E_c)-E0_c*np.conj(E0_c)
Ed1=np.real(Ed)
Ed_file = h5py.File('flux-out/E_diff_c1.h5','w')
Ed_file.create_dataset('ex',data=Ed1.T)
Ed_file.close()

Ed=E_c-E0_c
Ed1=np.real(Ed*np.conj(Ed))
Ed_file = h5py.File('flux-out/E_diff_c2.h5','w')
Ed_file.create_dataset('ex',data=Ed1.T)
Ed_file.close()

#plot subset of time-snapshots of the difference of electric fields
plt.figure()
plt.suptitle('Time stepped response')
for i in range(25):  
    print(int(i*120/24))
    plt.subplot(5,5,i+1)
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(Ed1[int(i*120/24)], interpolation='spline36', cmap='RdBu_r', alpha=0.9, vmin=Ed1.min(), vmax=Ed1.max())
    plt.ylabel('t= '+str(int(i*120/24)))
    plt.colorbar()
    #plt.axis('off')
plt.show()

showMaxSlice(Ed1,title='time response',cmap='RdBu')


