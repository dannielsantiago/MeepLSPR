#Import libraries
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import _shapes as shapes
from scipy.signal import find_peaks

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
nfreq = 100             # number of frequencies at which to compute flux
courant=0.5            # numerical stability, default is 0.5, should be lower in case refractive index n<1
time_step=0.05           # time step to measure flux
add_time=2             # additional time until field decays 1e-6
resolution =250        # resolution pixels/um (pixels/micrometers)
decay = 1e-12           # decay limit condition for the field measurement
cell = mp.Vector3(sx0, sy0, 0) 
monitor = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(mx,mx,0))

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
                     center=mp.Vector3(0,-sx,0),
                     amplitude=-1,
                     size=mp.Vector3(sw,0,0))

pt=mp.Vector3(1.1*rad,0,0) # 1.1*radpoint used to measure decay of Field (in oposite side of the source)



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
                    resolution=resolution,
                    Courant=courant)

#1.05112
#4.55112
flux_freqsXX=np.linspace(1.051120448179272,4.551120448179272,100)
dft_objx=[]
for i in range(nfreq):
    dft_objx=np.append(dft_objx,sim.add_dft_fields([polarisation], flux_freqsXX[i], flux_freqsXX[i], 1, where=monitor))
    
dft_obj = sim.add_dft_fields([mp.Ex], fcen, fcen, 1, where=monitor)

refl_t = sim.add_flux(fcen, df, nfreq, refl_fr_t) 
refl_b = sim.add_flux(fcen, df, nfreq, refl_fr_b)
refl_l = sim.add_flux(fcen, df, nfreq, refl_fr_l)
refl_r = sim.add_flux(fcen, df, nfreq, refl_fr_r)

sim.run(until_after_sources=mp.stop_when_fields_decayed(add_time,polarisation,pt,decay))

# for normalization run, save flux fields data for reflection plane
straight_refl_data_t = sim.get_flux_data(refl_t)
straight_refl_data_b = sim.get_flux_data(refl_b)
straight_refl_data_l = sim.get_flux_data(refl_l)
straight_refl_data_r = sim.get_flux_data(refl_r)

incident_flux = mp.get_fluxes(refl_b) 
incident_flux2 = mp.get_fluxes(refl_t) 

ex0_data = np.real(sim.get_dft_array(dft_obj, polarisation, 0))
flux_freqsX = np.array(mp.get_flux_freqs(refl_b))
ex0_data_array = []
for i in range(nfreq):
    ex0_data_array=np.append(ex0_data_array,np.real(sim.get_dft_array(dft_objx[i], polarisation, 0)))
ex0_data_array=ex0_data_array.reshape((nfreq,len(ex0_data),len(ex0_data)))


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

#
dft_objx=[]
for i in range(len(flux_freqsX)):
    dft_objx=np.append(dft_objx,sim.add_dft_fields([polarisation], flux_freqsX[i], flux_freqsX[i], 1, where=monitor))
    
dft_obj = sim.add_dft_fields([polarisation], fcen, fcen, 1, where=monitor)


sim.use_output_directory('flux-out_1')
sim.run(mp.in_volume(monitor, mp.at_beginning(mp.output_epsilon)),
        mp.in_volume(monitor, mp.to_appended("ez", mp.at_every(time_step, mp.output_efield_x))),
        until_after_sources=mp.stop_when_fields_decayed(add_time,polarisation,pt,decay))
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
flux_freqs = mp.get_flux_freqs(arefl_b)


eps_data = sim.get_array(vol=monitor, component=mp.Dielectric)
ex_data = np.real(sim.get_dft_array(dft_obj, polarisation, 0))

ex_data_array = []
for i in range(len(flux_freqsX)):
    ex_data_array=np.append(ex_data_array,np.real(sim.get_dft_array(dft_objx[i], polarisation, 0)))
ex_data_array=ex_data_array.reshape((nfreq,len(eps_data),len(eps_data)))


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
    
    norm = np.append(norm, incident_flux2[i]/max(incident_flux))
    tran = np.append(tran, transmitted_flux[i]/max(incident_flux))
    scatt = np.append(scatt, scat_refl_data_b[i]/max(incident_flux))

#multily for area (lenght in this case) of the sides to get the crossection in nm,
# area=4*rad=100 nm
scat=scat*4*rad*1000
abso=abso*4*rad*1000
ext=scat + abso

#peaks, _ = find_peaks(ext, height=max(ext)/2)

'''
---------------------------------
#calculate theoretical behaviour
check c_wire90.py for instructions
---------------------------------
'''
mat=theoretical.C_wire90(wl*1000,'Ag',1,shapes.rad*1000,16,'TM')

plt.figure()
#plt.plot(wl[peaks], ext[peaks], "x")
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
plt.axis([0.24, 0.7, 0, max(mat[0:,1])*1.2])
plt.grid(True)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
   
         
y=np.array(ext)[np.newaxis].T
yref=mat[0:,1]

deltaSignal = abs(y - yref)
percentageDifference = (deltaSignal/yref)*100 # Percent by element. *100
plt.figure()  
plt.plot(wl,percentageDifference, '--r', label='% error')
#plt.plot(wl,np.ones(len(wl))*meanPctDiff,'r', label='%.2f%% avg error' %meanPctDiff)
plt.axis([0.24, 0.7, 0, max(percentageDifference)*1.2])  
plt.grid(True)
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel("wavelength (um)")
plt.ylabel('% error', color='r')  
plt.tick_params('y', colors='r')
#plt.legend(loc="center right")  
plt.show()


plt.figure()
plt.subplot(3,2,1)
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(ex0_data.transpose(), interpolation='spline36', cmap='jet', alpha=0.9)

plt.subplot(3,2,2)
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(ex_data.transpose(), interpolation='spline36', cmap='jet', alpha=0.9)

plt.subplot(3,2,3)
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(ex_data.transpose()-ex0_data.transpose(), interpolation='spline36', cmap='jet', alpha=0.9)

plt.subplot(3,2,4)
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(np.square(ex_data.transpose()-ex0_data.transpose()), interpolation='spline36', cmap='jet', alpha=0.9)

plt.subplot(3,2,5)
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(np.log10(1/abs(ex_data.transpose()-ex0_data.transpose())), interpolation='spline36', cmap='jet', alpha=0.9)

plt.subplot(3,2,6)
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(np.log(np.square(ex_data.transpose()-ex0_data.transpose())), interpolation='spline36', cmap='jet', alpha=0.9)

plt.axis('off')
plt.show()
#FT with particle
plt.figure()
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(np.square(ex_data_array[int(99-(i*99/24))].transpose()), interpolation='spline36', cmap='jet', alpha=0.9)
    plt.ylabel(str(int(1000/flux_freqs[int(99-(i*99/24))]))+' nm')
plt.show()
#ft without particle
plt.figure()
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(np.square(ex0_data_array[int(99-(i*99/24))].transpose()), interpolation='spline36', cmap='jet', alpha=0.9)
    plt.ylabel(str(int(1000/flux_freqs[int(99-(i*99/24))]))+' nm')
plt.show()

data2=[]
datalog=[]
for i in range(25):
    data2=np.append(data2,np.square(ex_data_array[int(99-(i*99/24))].transpose()-ex0_data_array[int(99-(i*99/24))].transpose()))
    datalog=np.append(datalog,np.log10(np.square(ex_data_array[int(99-(i*99/24))].transpose()-ex0_data_array[int(99-(i*99/24))].transpose())))
data2=data2.reshape((25,len(eps_data),len(eps_data)))
datalog=datalog.reshape((25,len(eps_data),len(eps_data)))


plt.figure()
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(data2[i], interpolation='spline36', cmap='jet', alpha=0.9, vmin=data2.min(), vmax=data2.max())
    plt.ylabel(str(int(1000/flux_freqs[int(99-(i*99/24))]))+' nm')
    plt.colorbar()
plt.show()


plt.figure()
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(datalog[i], interpolation='spline36', cmap='jet', alpha=0.9, vmin=datalog.min(), vmax=datalog.max())
    plt.ylabel(str(int(1000/flux_freqs[int(99-(i*99/24))]))+' nm')
    plt.colorbar()
plt.show()

'''

import h5py
hf = h5py.File('data.h5', 'a')
hf.create_dataset('wl', data=wl)
g1 = hf.create_group('45nm')
g1.create_dataset('ext', data=ext)
g1.create_dataset('scat', data=scat)
g1.create_dataset('abso', data=abso)
g1.create_dataset('A_ext', data=mat[0:,1])
g1.create_dataset('A_scat', data=mat[0:,2])
g1.create_dataset('A_abso', data=mat[0:,3])
hf.close()



plot error
  
eps_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(mx,mx,0), component=mp.Dielectric)
ez_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(mx,mx,0), component=mp.Ez)
plt.figure()
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
plt.axis('off')
plt.show()   

'''

'''
#upper=np.array([mat[0:,1][i]+percentageDifference[i] for i in range(len(ext))]).flatten()
#lower=np.array([mat[0:,1][i]-percentageDifference[i] for i in range(len(ext))]).flatten()
from scipy.stats import sem
sem1=sem(yref)
upper=np.array([mat[0:,1][i]+2*sem1 for i in range(len(ext))]).flatten()
lower=np.array([mat[0:,1][i]-2*sem1 for i in range(len(ext))]).flatten()



#plt.plot(wl,Ag25nmTMext,'o', label='ext 25nm')
#plt.plot(wl,Ag35nmTMext,'s', label='ext 35nm')
#plt.plot(wl,ext,'^', label='ext 45nm')
#plt.plot(mat[0:,0],Ag25nmTMexttheory, '-k', label='Analytical model')
#plt.plot(mat[0:,0],Ag35nmTMextTheory, '-k')
#plt.plot(mat[0:,0],mat[0:,1], '-k')
#plt.fill_between(wl, lower, upper, facecolor='red', alpha=0.5,interpolate=True)

scat_t = np.append(scat_t, abs(scat_refl_data_t[i]/incident_flux[i]))
    scat_b = np.append(scat_b, abs(scat_refl_data_b[i]/incident_flux[i]))
    scat_l = np.append(scat_l, abs(scat_refl_data_l[i]/incident_flux[i]))
    scat_r = np.append(scat_r, abs(scat_refl_data_r[i]/incident_flux[i]))

    abso_t = np.append(abso_t, abs(abs_refl_data_t[i]/incident_flux[i]))
    abso_b = np.append(abso_b, abs(abs_refl_data_b[i]/incident_flux[i]))
    abso_l = np.append(abso_l, abs(abs_refl_data_l[i]/incident_flux[i]))
    abso_r = np.append(abso_r, abs(abs_refl_data_r[i]/incident_flux[i]))
    
#plt.plot(wl,abso_t/max(ext),'^r')
#plt.plot(wl,abso_b/max(ext),'vr')
#plt.plot(wl,abso_l/max(ext),'<r')
#plt.plot(wl,abso_r/max(ext),'>r')

#plt.plot(wl,scat_t/max(ext),'^b')
#plt.plot(wl,scat_b/max(ext),'vb')
#plt.plot(wl,scat_l/max(ext),'<b')
#plt.plot(wl,scat_r/max(ext),'>b')

'''