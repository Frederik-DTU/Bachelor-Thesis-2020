# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:22:30 2020

@author: Frederik
"""

import numpy as np
from scipy.optimize import minimize

from sphere_statistics import sphere_statistics
import numpy as np

from scipy.io import loadmat
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import os
import platform
import sys

#%% Problem

def avg_theta(data):
    n = data.shape[1]
    
    sum1 = 0
    
    for i in range(0,n):
        sum1 += np.arccos(data[2,i])
        
    avg = sum1/n
    
    return avg

def rot_data(x, data):
    Rx = np.array([[1,0,0],
                   [0,np.cos(x[0]), -np.sin(x[0])],
                   [0,np.sin(x[0]), np.cos(x[0])]])
        
    Ry = np.array([[np.cos(x[1]),0, np.sin(x[1])],
                   [0, 1, 0],
                   [-np.sin(x[1]), 0, np.cos(x[1])]])
    
    #Rz = np.array([[np.cos(x[2]), -np.sin(x[2]),0],
    #               [np.sin(x[2]), np.cos(x[2]), 0],
    #               [0, 0, 1]])
    
    n = data.shape[1]
    
    sum1 = 0
    sum2 = 0
    
    data_rot = data
    
    #Rinv = np.linalg.inv( Rx @ Ry @ Rz)
    Rinv = np.linalg.inv( Rx @ Ry)
    
    for i in range(0,n):
        data_rot[:,i] = Rinv @ data[:,i]
    
    return data_rot

def rot_matrix(x):
    Rx = np.array([[1,0,0],
                   [0,np.cos(x[0]), -np.sin(x[0])],
                   [0,np.sin(x[0]), np.cos(x[0])]])
        
    Ry = np.array([[np.cos(x[1]),0, np.sin(x[1])],
                   [0, 1, 0],
                   [-np.sin(x[1]), 0, np.cos(x[1])]])
    
    #Rz = np.array([[np.cos(x[2]), -np.sin(x[2]),0],
    #               [np.sin(x[2]), np.cos(x[2]), 0],
    #               [0, 0, 1]])
    
    #return Rx @ Ry @ Rz

    return Rx @ Ry

def object_fun(x, data):
    Rx = np.array([[1,0,0],
                   [0,np.cos(x[0]), -np.sin(x[0])],
                   [0,np.sin(x[0]), np.cos(x[0])]])
        
    Ry = np.array([[np.cos(x[1]),0, np.sin(x[1])],
                   [0, 1, 0],
                   [-np.sin(x[1]), 0, np.cos(x[1])]])
    
    #Rz = np.array([[np.cos(x[2]), -np.sin(x[2]),0],
    #               [np.sin(x[2]), np.cos(x[2]), 0],
    #               [0, 0, 1]])
    
    n = data.shape[1]
    
    sum1 = 0
    sum2 = 0
    
    for i in range(0,n):
        #data_rot = Rx @ Ry @ Rz @ data[:,i]
        data_rot = Rx @ Ry @ data[:,i]
        theta_rot = np.arccos(data_rot[2])
        sum1 += theta_rot*theta_rot
        sum2 += theta_rot
        
    object_val = sum1/n+sum2*sum2/(n*n)
    
    return object_val

b = (0, 2*np.pi)
bnds = (b, b, b)
x0 = [0,0,0]
#x0 = [1,1,1]

bnds = (b, b)
x0 = [0,0]

#%% Solution for data

sys_name = platform.system()
if sys_name == "Darwin":
    #Loading data - Mac
    path = '/Users/Frederik/CloudStation/DTU/Bachelor - Matematik & Teknologi/Semester 6/Bachelorprojekt/Programmer og Data/Data/'
    file_path = path + '/sphere_walking.mat'
elif sys_name == "Windows":
    # Loading data - Windows
    path = 'E:\\Frederik\\CloudStation\\DTU\\Bachelor - Matematik & Teknologi\\Semester 6\\Bachelorprojekt\\Programmer og Data\\Data'
    file_path = path + '\\sphere_walking.mat'
else:
    print("\nUnkown system platform. Ending the program!\n")
    sys.exit()

knee_sphere = loadmat(file_path, squeeze_me=True)
raw_data = knee_sphere['data']
t = np.linspace(-0.5,0.5, 100)

data_num = raw_data.shape[1] #Number of data points


sol = minimize(object_fun, x0, method = 'SLSQP', bounds = bnds, args = (raw_data))

#%% Plotting the solution

theta_c = avg_theta(raw_data)
data_rot = rot_data(sol.x, raw_data)

t = np.linspace(-np.pi, np.pi, 100)

knee_data = sphere_statistics(data_type = "Own", data = raw_data, n = data_num)

param = np.zeros([3,100])
param[0,:] = np.sin(theta_c)*np.cos(t)
param[1,:] = np.sin(theta_c)*np.sin(t)
param[2,:] = np.cos(theta_c)

R = rot_matrix(sol.x)
Rinv = np.linalg.inv(R)

param_rot = Rinv @ param

data_tangent, param_ts, param_rot_ts = knee_data.tangent_space(np.array([0,0,1]), raw_data, 
                                                                param, param_rot)

fig = plt.figure(figsize = (5,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.scatter(data_tangent[0], data_tangent[1], color = 'black', 
            label = 'Data points')
plt.plot(param_ts[0], param_ts[1], 
         label = 'Optimal circle parallel to the xy-plane', color = 'red')
plt.plot(param_rot_ts[0], param_rot_ts[1], 
         label = 'Optimal circle', color = 'blue')

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend()

# Plotting the data 
fig = plt.figure(figsize = (5,5))
ax = plt.axes(projection='3d')
col_val = np.linspace(start = 0, stop = 0, num = data_num) #Set start=stop for same color

ax.scatter3D(raw_data[0], raw_data[1], raw_data[2], color = 'Black', label = 'Data points');

#p = ax.scatter3D(data_rot[0], data_rot[1], data_rot[2], c = 'orange');

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.tight_layout()

# Adding a sphere (S2) to the data
r = 1
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)

ax.plot(param_rot[0], param_rot[1], param_rot[2], 
        label = 'Optimal Circle', c = 'blue')

ax.plot(param[0], param[1], param[2], 
        label = 'Optimal circle parallel to the xy-plane', c = 'red')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()

plt.show()

circle_mean = Rinv @ np.array([0,0,1])

#%% Karcher mean for data

knee_data = sphere_statistics(data_type = "Own", data = raw_data, n = data_num)

mean_val = knee_data.intrinsic_mean_gradient_search(tau=1)

data_tangent, circle_mean_ts, param_rot_ts = knee_data.tangent_space(mean_val[:,-1], raw_data,
                                                                    circle_mean,
                                                                    param_rot)

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val.shape[1]) #Set start=stop for same color

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.scatter(data_tangent[0], data_tangent[1], color = 'black', 
            label = 'Data points')
plt.scatter(0,0,
         label = 'Karcher mean', color = 'blue', s = 100)
plt.scatter(circle_mean_ts[0], circle_mean_ts[1], 
         label = 'Center of optimal circle', color = 'red', s = 100)
plt.plot(param_rot_ts[0], param_rot_ts[1], 
         label = 'Optimal circle', color = 'blue')

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend(loc = 'upper right')

# Plotting the data 
fig = plt.figure(figsize = (5,5))
ax = plt.axes(projection='3d')

ax.scatter3D(raw_data[0], raw_data[1], raw_data[2], color = 'black', label = 'Data points')

ax.scatter3D(mean_val[:,-1][0], mean_val[:,-1][1], mean_val[:,-1][2], c = 'blue', 
                 s = 100, label = 'Karcher mean')

ax.scatter3D(circle_mean[0], circle_mean[1], circle_mean[2], c = 'red', 
                 s = 100, label = 'Mean of the optimal circle')

ax.plot(param_rot[0], param_rot[1], param_rot[2], 
        label = 'Optimal Circle', c = 'blue')

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.tight_layout()

# Adding a sphere (S2) to the data
r = 1
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()

plt.show()

#%% Testing for other initial guesses

for guess in [[0,0],[0.5,0.5],[1,1],[1.5,1.5],[2,2]]:
    check = minimize(object_fun, guess, method = 'SLSQP', bounds = bnds, args = (raw_data))
    print(check.fun)
    
    t = np.linspace(-np.pi, np.pi, 100)
    
    param = np.zeros([3,100])
    param[0,:] = np.sin(theta_c)*np.cos(t)
    param[1,:] = np.sin(theta_c)*np.sin(t)
    param[2,:] = np.cos(theta_c)
    
    R = rot_matrix(check.x)
    Rinv = np.linalg.inv(R)
    
    param_rot = Rinv @ param
    
    fig = plt.figure(figsize = (5,5))
    ax = plt.axes(projection='3d')
    col_val = np.linspace(start = 0, stop = 0, num = data_num) #Set start=stop for same color
    
    ax.scatter3D(raw_data[0], raw_data[1], raw_data[2], color = 'Black', label = 'Data points');
    
    #p = ax.scatter3D(data_rot[0], data_rot[1], data_rot[2], c = 'orange');
    
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    plt.tight_layout()
    
    # Adding a sphere (S2) to the data
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)
    
    ax.plot(param_rot[0], param_rot[1], param_rot[2], 
            label = 'Optimal Circle', c = 'blue')
    
    ax.plot(param[0], param[1], param[2], 
            label = 'Optimal circle parallel to the xy-plane', c = 'red')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.legend()
    
    plt.show()





            