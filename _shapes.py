#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 12:35:10 2019
Different shapes to be used by the plasmonic structure
@author: r2d2
"""

#Import libraries
import meep as mp
from materials_library import Ag, Au, cSi, myAg
#import _materials as lib
import math


# A cylinder of infinite radius and height 0.25 pointing along the x axis,
# centered at the origin:
rad=0.025 #micrometers units
material=Ag

cyl = mp.Cylinder(center=mp.Vector3(0,0,0), height=mp.inf, radius=rad,
                  axis=mp.Vector3(0,0,1), material=material)

cyl2 = mp.Cylinder(center=mp.Vector3(0,0,0), height=mp.inf, radius=rad,
                  axis=mp.Vector3(0,0,1), material=mp.Medium(index=5))

water = mp.Cylinder(center=mp.Vector3(0,0,0), height=mp.inf, radius=rad+0.005,
                  axis=mp.Vector3(0,0,1), material=mp.Medium(epsilon=1.69))

sph = mp.Sphere(radius=rad, center=mp.Vector3(0,0,0), material=Ag)

# An ellipsoid with its long axis pointing along (1,1,1), centered on
# the origin (the other two axes are orthogonal and have equal semi-axis lengths):
ell = mp.Ellipsoid(center=mp.Vector3(0,0,0), size=mp.Vector3(0.02,0.03,0.01),
                   e1=mp.Vector3(1,0,0), e2=mp.Vector3(0,1,0), e3=mp.Vector3(0,0,1),
                   material=Ag )

# A unit cube of material metal with a spherical air hole of radius 0.2 at
# its center, the whole thing centered at (1,2,3):
gBlock=[mp.Block(center=mp.Vector3(1,2,3), size=mp.Vector3(1,1,1), material=mp.metal),
          mp.Sphere(center=mp.Vector3(1,2,3), radius=0.2, material=mp.air)]

# A hexagonal prism defined by six vertices centered on the origin
# of material crystalline silicon (from the materials library)
vertices = [mp.Vector3(-1,0),
            mp.Vector3(-0.5,math.sqrt(3)/2),
            mp.Vector3(0.5,math.sqrt(3)/2),
            mp.Vector3(1,0),
            mp.Vector3(0.5,-math.sqrt(3)/2),
            mp.Vector3(-0.5,-math.sqrt(3)/2)]
geometryP = [mp.Prism(vertices, height=1.5, center=mp.Vector3(), material=cSi)]
'''
(define myAg (make dielectric (epsilon 1)
(polarizations
(make polarizability
(omega 1e-20) (gamma 0.0038715) (sigma 4.4625e+39))
(make polarizability
(omega 0.065815) (gamma 0.31343) (sigma 7.9247))
(make polarizability
(omega 0.36142) (gamma 0.036456) (sigma 0.50133))
(make polarizability
(omega 0.66017) (gamma 0.0052426) (sigma 0.013329))
(make polarizability
(omega 0.73259) (gamma 0.07388) (sigma 0.82655))
(make polarizability
(omega 1.6365) (gamma 0.19511) (sigma 1.1133))
)))
'''