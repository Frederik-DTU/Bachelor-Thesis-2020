# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 00:31:07 2020

@author: Frederik
"""

import numpy as np

from scipy.io import loadmat
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sphere_statistics import sphere_statistics

import os
import platform
import sys

#Plot error
fig = plt.figure(figsize=(6,5))
ax = plt.axes(projection='3d')

r = 1
pi = np.pi
cos = np.cos
sin = np.sin
c, theta = np.mgrid[0.0:pi/2:100j, 0.0:pi/2:100j]
x = c
y = theta
z = np.abs(c*cos(theta)-np.arccos(cos(c)/(np.sqrt(1-(sin(c)*sin(theta))**2))))

ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='c', alpha=1, linewidth=0)

ax.set_xlabel('c')
ax.set_ylabel('theta')
ax.set_zlabel('Error function')

ax.legend()

plt.show()

#Plot error
fig = plt.figure(figsize=(6,5))
ax = plt.axes(projection='3d')

r = 1
pi = np.pi
cos = np.cos
sin = np.sin
c, theta = np.mgrid[0.0:pi/4:100j, 0.0:pi/2:100j]
x = c
y = theta
z = np.abs(c*cos(theta)-np.arccos(cos(c)/(np.sqrt(1-(sin(c)*sin(theta))**2))))

ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='c', alpha=1, linewidth=0)

ax.set_xlabel('c')
ax.set_ylabel('theta')
ax.set_zlabel('Error function')

ax.legend()

plt.show()