# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 01:58:43 2020

@author: Frederik
"""

from sphere_statistics import sphere_statistics
import numpy as np

from scipy.io import loadmat
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import os
import platform
import sys

#%% Karcher mean for the real data

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

data_num = raw_data.shape[1] #Number of data points

knee_data = sphere_statistics(data_type = "Own", data = raw_data, n = data_num)

mean_val = knee_data.intrinsic_mean_gradient_search(tau=1)

data_tangent, mean_val_tangent_space = knee_data.tangent_space(mean_val[:,-1], raw_data, 
                                                                mean_val)

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val.shape[1]) #Set start=stop for same color

fig = plt.figure(figsize = (5,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.scatter(data_tangent[0], data_tangent[1], color = 'black', 
            label = 'Data points')
p = plt.scatter(mean_val_tangent_space[0], mean_val_tangent_space[1], c = col_val,
                s = 100*np.ones(mean_val.shape[1]),
            alpha = 0.4)

fig.colorbar(p)

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

# Plotting the data 
fig = plt.figure(figsize=(6,5))
ax = plt.axes(projection='3d')

p = ax.scatter3D(raw_data[0], raw_data[1], raw_data[2], color = 'black',
                 label = 'Data points')

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val.shape[1]) #Set start=stop for same color


p = ax.scatter3D(mean_val[0], mean_val[1], mean_val[2], c = col_val, 
                 s = 100*np.ones(mean_val.shape[1]))

fig.colorbar(p)

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



#%% Karcher mean for simulated data on geodesic curve

data_num = 3
p1 = np.array([0,0,1])
p2 = np.array([np.sqrt(2)/2, np.sqrt(2)/2,0])

curve = sphere_statistics(data_type = "Geodesic_curve", n = data_num, p = p1, q = p2)
data = curve.get_data()
geodesic_line = curve.sim_geodesic(100, p1, p2)

mean_val = curve.intrinsic_mean_gradient_search(tau = 1)

data_tangent, geodesic_line_tangent_space, mean_val_tangent_space = curve.tangent_space(np.array([1/2,1/2,np.sqrt(2)/2]), data, 
                                                                geodesic_line, mean_val)

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val.shape[1]) #Set start=stop for same color

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.scatter(data_tangent[0], data_tangent[1], color = 'black', 
            label = 'Data points')
plt.plot(geodesic_line_tangent_space[0], geodesic_line_tangent_space[1], 
         label = 'Geodesic line', color = 'blue')
p = plt.scatter(mean_val_tangent_space[0], mean_val_tangent_space[1], c = col_val,
                s = 100*np.ones(mean_val.shape[1]),
            alpha = 0.4)

fig.colorbar(p)

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

# Plotting the data 
fig = plt.figure(figsize=(6,5))
ax = plt.axes(projection='3d')

p = ax.scatter3D(data[0], data[1], data[2], color = 'black', label = 'Data points')

p = ax.scatter3D(mean_val[0], mean_val[1], mean_val[2], c = col_val, 
                 s = 100*np.ones(mean_val.shape[1]), alpha = 0.4)

fig.colorbar(p)

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

ax.plot(geodesic_line[0,:], geodesic_line[1,:], geodesic_line[2,:], color = 'blue',
        label = 'Geodesic line')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()

plt.show()

curve = sphere_statistics(data_type = "Geodesic_curve", n = 10, p = p1, q = p2)
data = curve.get_data()
geodesic_line = curve.sim_geodesic(100, p1, p2)

mean_val = curve.intrinsic_mean_gradient_search(tau = 1)

data_tangent, geodesic_line_tangent_space, mean_val_tangent_space = curve.tangent_space(np.array([1/2,1/2,np.sqrt(2)/2]), data, 
                                                                geodesic_line, mean_val)

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val.shape[1]) #Set start=stop for same color

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.scatter(data_tangent[0], data_tangent[1], color = 'black', 
            label = 'Data points')
plt.plot(geodesic_line_tangent_space[0], geodesic_line_tangent_space[1], 
         label = 'Geodesic line', color = 'blue')
p = plt.scatter(mean_val_tangent_space[0], mean_val_tangent_space[1], c = col_val,
                s = 100*np.ones(mean_val.shape[1]),
            alpha = 0.4)

fig.colorbar(p)

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

# Plotting the data 
fig = plt.figure(figsize=(6,5))
ax = plt.axes(projection='3d')

p = ax.scatter3D(data[0], data[1], data[2], color = 'black', label = 'Data points')

p = ax.scatter3D(mean_val[0], mean_val[1], mean_val[2], c = col_val, 
                 s = 100*np.ones(mean_val.shape[1]), alpha = 0.4)

fig.colorbar(p)

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

ax.plot(geodesic_line[0,:], geodesic_line[1,:], geodesic_line[2,:], color = 'blue',
        label = 'Geodesic line')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()

plt.show()

p_end = np.array([np.sqrt(2)/2, np.sqrt(2)/2,0])
p_end.shape = (3,1)
curve = sphere_statistics(data_type = "Own", 
                          data = np.concatenate((data[:,0:5],p_end), axis = 1),
                          n = 6)

data = curve.get_data()
geodesic_line = curve.sim_geodesic(100, p1, p2)

mean_val = curve.intrinsic_mean_gradient_search(tau = 1)

data_tangent, geodesic_line_tangent_space, mean_val_tangent_space = curve.tangent_space(np.array([0.3712, 0.3712, 0.8511]), 
                                                                                        data, 
                                                                geodesic_line, mean_val)

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val.shape[1]) #Set start=stop for same color

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.scatter(data_tangent[0], data_tangent[1], color = 'black', 
            label = 'Data points')
plt.plot(geodesic_line_tangent_space[0], geodesic_line_tangent_space[1], 
         label = 'Geodesic line', color = 'blue')
p = plt.scatter(mean_val_tangent_space[0], mean_val_tangent_space[1], c = col_val,
                s = 100*np.ones(mean_val.shape[1]),
            alpha = 0.4)

fig.colorbar(p)

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

# Plotting the data 
fig = plt.figure(figsize=(6,5))
ax = plt.axes(projection='3d')

p = ax.scatter3D(data[0], data[1], data[2], color = 'black', label = 'Data points')

p = ax.scatter3D(mean_val[0], mean_val[1], mean_val[2], c = col_val, 
                 s = 100*np.ones(mean_val.shape[1]), alpha = 0.4)

fig.colorbar(p)

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

ax.plot(geodesic_line[0,:], geodesic_line[1,:], geodesic_line[2,:], color = 'blue',
        label = 'Geodesic line')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()

plt.show()

#%% Karcher mean for simulated data on geodesic ball

data_num = 10
p1 = np.array([0,0,1])
radius = 1

ball = sphere_statistics(data_type = "Geodesic_ball", n = data_num, p = p1, r = radius)
data = ball.get_data()

mean_val = ball.intrinsic_mean_gradient_search(tau = 1)

ball_boundary = sphere_statistics(data_type = "Geodesic_ball", n = 100, p = p1, r = radius)
data_boundary = ball_boundary.get_data()

data_tangent, mean_val_tangent_space, data_boundary_ts = ball.tangent_space(np.array([0,0,1]), data, 
                                                                            mean_val, 
                                                                            data_boundary)

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val.shape[1]) #Set start=stop for same color

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

scatter = plt.scatter(data_tangent[0], data_tangent[1], 
            label = 'Data points', color = 'black')
plt.plot(data_boundary_ts[0], data_boundary_ts[1], 
         label = 'Boundary of geodesic ball', color = 'blue')
p = plt.scatter(mean_val_tangent_space[0], mean_val_tangent_space[1], c = col_val,
                s = 100*np.ones(mean_val.shape[1]),
            alpha = 0.4)

fig.colorbar(p)

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)


# Plotting the data 
fig = plt.figure(figsize=(6,5))
ax = plt.axes(projection='3d')

p = ax.scatter3D(data[0], data[1], data[2], color = 'black', label = 'Data points')

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val.shape[1]) #Set start=stop for same color


p = ax.scatter3D(mean_val[0], mean_val[1], mean_val[2], c = col_val, 
                 s = 100*np.ones(mean_val.shape[1]))

fig.colorbar(p)

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

ax.plot(data_boundary[0,:], data_boundary[1,:], data_boundary[2,:], color = 'blue',
        label = 'Boundary of geodesic ball')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()

plt.show()


#%% Karcher mean for the north and south pole

guess1 = np.array([np.sqrt(2)/2,0,np.sqrt(2)/2])
guess2 = np.array([0,np.sqrt(2)/2,np.sqrt(2)/2])
data = np.array([[0,0],[0,0], [1,-1]])

data_num = data.shape[1] #Number of data points
t = np.linspace(0,1,100)

north_south = sphere_statistics(data_type = "Own", data = data, n = data_num)

mean_val1 = north_south.intrinsic_mean_gradient_search(tau=1, initial_guess = guess1)
mean_val2 = north_south.intrinsic_mean_gradient_search(tau=1, initial_guess = guess2)

geodesic_line1 = north_south.geodesic_line(t, data[:,0], guess1)
geodesic_line2 = north_south.geodesic_line(t, data[:,0], guess2)

data_tangent1, geodesic_line1_ts1, geodesic_line2_ts1, mean_val1_ts1, mean_val2_ts1 = ball.tangent_space(np.array([1,0,0]), 
                                                                           data, 
                                                                            geodesic_line1, 
                                                                            geodesic_line2,
                                                                            mean_val1,
                                                                            mean_val2)
data_tangent2, geodesic_line1_ts2, geodesic_line2_ts2, mean_val1_ts2, mean_val2_ts2 = ball.tangent_space(np.array([0,1,0]), 
                                                                           data, 
                                                                            geodesic_line1, 
                                                                            geodesic_line2,
                                                                            mean_val1,
                                                                            mean_val2)

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val1_ts1.shape[1]) #Set start=stop for same color

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

scatter = plt.scatter(data_tangent1[0], data_tangent1[1], 
            label = 'Data points', color = 'black')
plt.plot(geodesic_line1_ts1[0], geodesic_line1_ts1[1], 
         label = 'Geodesic curve', color = 'blue')
plt.plot(geodesic_line2_ts1[0], geodesic_line2_ts1[1], 
         label = 'Geodesic curve', color = 'red')
p = plt.scatter(mean_val1_ts1[0], mean_val1_ts1[1], c = col_val,
                s = 100*np.ones(mean_val1_ts1.shape[1]),
            alpha = 0.4)

p = plt.scatter(mean_val1_ts2[0], mean_val2_ts1[1], c = col_val,
                s = 100*np.ones(mean_val2_ts1.shape[1]),
            alpha = 0.4)

fig.colorbar(p)

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

scatter = plt.scatter(data_tangent2[0], data_tangent2[1], 
            label = 'Data points', color = 'black')
plt.plot(geodesic_line1_ts2[0], geodesic_line1_ts2[1], 
         label = 'Geodesic curve', color = 'blue')
plt.plot(geodesic_line2_ts2[0], geodesic_line2_ts2[1], 
         label = 'Geodesic curve', color = 'red')
p = plt.scatter(mean_val1_ts2[0], mean_val1_ts2[1], c = col_val,
                s = 100*np.ones(mean_val1_ts2.shape[1]),
            alpha = 0.4)

p = plt.scatter(mean_val1_ts2[0], mean_val2_ts2[1], c = col_val,
                s = 100*np.ones(mean_val2_ts2.shape[1]),
            alpha = 0.4)

fig.colorbar(p)

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)


# Plotting the data 
fig = plt.figure(figsize=(6,5))
ax = plt.axes(projection='3d')

p = ax.scatter3D(data[0], data[1], data[2], color = 'black', label = 'Data points')

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val1.shape[1]) #Set start=stop for same color


p = ax.scatter3D(mean_val1[0], mean_val1[1], mean_val1[2], c = ['purple', 'yellow'], 
                 s = 100*np.ones(mean_val1.shape[1]))

fig.colorbar(p)

p = ax.scatter3D(mean_val2[0], mean_val2[1], mean_val2[2], c = ['purple', 'yellow'], 
                 s = 100*np.ones(mean_val2.shape[1]))

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

ax.plot(geodesic_line1[0,:], geodesic_line1[1,:], geodesic_line1[2,:], c = 'blue', 
        label = 'Geodesic curve')
ax.plot(geodesic_line2[0,:], geodesic_line2[1,:], geodesic_line2[2,:], c = 'red',
        label = 'Geodesic curve')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()

plt.show()

#%% Karcher mean for simulated data on geodesic ball

guess1 = np.array([0,np.sqrt(2)/2,-np.sqrt(2)/2])
guess2 = np.array([0,np.sqrt(2)/2,np.sqrt(2)/2])
data_num = 10
p1 = np.array([0,0,1])
radius = np.pi/2

ball = sphere_statistics(data_type = "Geodesic_ball", n = data_num, p = p1, r = radius)
data = ball.get_data()

ball_boundary = sphere_statistics(data_type = "Geodesic_ball", n = 100, p = p1, r = radius)
data_boundary = ball_boundary.get_data()

mean_val1 = ball.intrinsic_mean_gradient_search(tau = 1, initial_guess = guess1)
mean_val2 = ball.intrinsic_mean_gradient_search(tau = 1, initial_guess = guess2)

data_tangent1, mean_val_ts1, data_boundary_ts1 = ball.tangent_space(np.array([0,0,-1]), data, 
                                                                            mean_val1, 
                                                                            data_boundary)

data_tangent2, mean_val_ts2, data_boundary_ts2 = ball.tangent_space(np.array([0,0,1]), data, 
                                                                            mean_val2, 
                                                                            data_boundary)

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val1.shape[1]) #Set start=stop for same color

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

scatter = plt.scatter(data_tangent1[0], data_tangent1[1], 
            label = 'Data points', color = 'black')
plt.plot(data_boundary_ts1[0], data_boundary_ts1[1], 
         label = 'Equator', color = 'blue')
p = plt.scatter(mean_val_ts1[0], mean_val_ts1[1], c = col_val,
                s = 100*np.ones(mean_val_ts1.shape[1]),
            alpha = 0.4)

fig.colorbar(p)

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

scatter = plt.scatter(data_tangent2[0], data_tangent2[1], 
            label = 'Data points', color = 'black')
plt.plot(data_boundary_ts2[0], data_boundary_ts2[1], 
         label = 'Equator', color = 'blue')
p = plt.scatter(mean_val_ts2[0], mean_val_ts2[1], c = col_val,
                s = 100*np.ones(mean_val_ts2.shape[1]),
            alpha = 0.4)

fig.colorbar(p)

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

# Plotting the data 
fig = plt.figure(figsize=(6,5))
ax = plt.axes(projection='3d')

p = ax.scatter3D(data[0], data[1], data[2], color = 'black', label = 'Data points')

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val1.shape[1]) #Set start=stop for same color


p = ax.scatter3D(mean_val1[0], mean_val1[1], mean_val1[2], c = col_val, 
                 s = 100*np.ones(mean_val1.shape[1]))

fig.colorbar(p)

p = ax.scatter3D(mean_val2[0], mean_val2[1], mean_val2[2], c = col_val, 
                 s = 100*np.ones(mean_val2.shape[1]))

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

ax.plot(data_boundary[0,:], data_boundary[1,:], data_boundary[2,:], color = 'blue',
        label = 'Equator')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()

plt.show()





