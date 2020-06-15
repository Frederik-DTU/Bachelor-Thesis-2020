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
t = np.linspace(-0.5,0.5, 100)

data_num = raw_data.shape[1] #Number of data points

knee_data = sphere_statistics(data_type = "Own", data = raw_data, n = data_num)

mean_val = knee_data.intrinsic_mean_gradient_search(tau=1)

pga = knee_data.pga(mean_val[:,-1])
geodesic_line1 = knee_data.geodesic_line(t, mean_val[:,-1],pga['v_k_exp'][:,0])
geodesic_line2 = knee_data.geodesic_line(t, mean_val[:,-1],pga['v_k_exp'][:,1])

# Compute variance explained by principal components
rho = (pga['lambda']*pga['lambda']) / (pga['lambda']*pga['lambda']).sum()
rho = rho[:-1]  

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal geodesics');
plt.xlabel('Principal geodesics');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.ylim([0,1.05])
plt.show()

data_tangent, geodesic_line_ts1, geodesic_line_ts2, mean_val_ts = knee_data.tangent_space(mean_val[:,-1], 
                                                                                          raw_data, 
                                                                geodesic_line1, geodesic_line2,
                                                                mean_val)

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.scatter(data_tangent[0], data_tangent[1], color = 'black', 
            label = 'Data points')
plt.plot(geodesic_line_ts1[0], geodesic_line_ts1[1], 
         label = '1. PG ('+ str(100*rho[0])[0:4] + '%)', color = 'blue')
if rho[1]<0.001:
    plt.plot(geodesic_line_ts2[0], geodesic_line_ts2[1], 
         label = '2. PG ('+ str(0) + '%)', color = 'orange')
else:
    plt.plot(geodesic_line_ts2[0], geodesic_line_ts2[1], 
         label = '2. PG ('+ str(100*rho[1])[:4] + '%)', color = 'orange')
p = plt.scatter(mean_val_ts[0,-1], mean_val_ts[1,-1], color = 'blue',
                s = 100*np.ones(mean_val.shape[1]), label = 'Karcher mean',
            alpha = 0.4)

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend()

# Plotting the data 
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')

p = ax.scatter3D(raw_data[0], raw_data[1], raw_data[2], color = 'black', label = 'Data points')

p = ax.scatter3D(mean_val[0,-1], mean_val[1,-1], mean_val[2,-1],s=100, color = 'blue',
                 label = 'Karcher mean')

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

ax.plot(geodesic_line1[0,:], geodesic_line1[1,:], geodesic_line1[2,:], 
        label = '1. PG (' + str(100*rho[0])[0:4] + '%)')
if rho[1]<0.001:
    ax.plot(geodesic_line2[0,:], geodesic_line2[1,:], geodesic_line2[2,:], 
            label = '2. PG (' + str(0) + '%)')
else:
    ax.plot(geodesic_line2[0,:], geodesic_line2[1,:], geodesic_line2[2,:], 
            label = '2. PG (' + str(100*rho[1])[0:4] + '%)')

# Put a legend below current axis
ax.legend()

plt.show()

pca = knee_data.pca(mean_val[:,-1])

# Compute variance explained by principal components
rho = (pca['sigma']*pca['sigma']) / (pca['sigma']*pca['sigma']).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# Plotting the data 
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

origin = pca['mean'][0], pca['mean'][1], pca['mean'][2]
p = ax.scatter3D(pca['mean'][0], pca['mean'][1], pca['mean'][2], color = 'blue',
                label = 'Arithmetic Mean', s = 100,
                alpha = 0.4)
plt.quiver(*origin, pca['v_k'][0,0],
           pca['v_k'][1,0], pca['v_k'][2,0], color = 'blue', label = '1. PC (' 
           + str(100*rho[0])[0:4] + '%)')
if rho[1]<0.001:
    plt.quiver(*origin, pca['v_k'][0,1],
               pca['v_k'][1,1], pca['v_k'][2,1], color = 'orange', label = '2. PC ('
               + str(0) + '%)')
else:
    plt.quiver(*origin, pca['v_k'][0,1],
               pca['v_k'][1,1], pca['v_k'][2,1], color = 'orange', label = '2. PC ('
               + str(100*rho[1])[0:4] + '%)')
if rho[2] < 0.001:    
    plt.quiver(*origin, pca['v_k'][0,2],
               pca['v_k'][1,2], pca['v_k'][2,2], color = 'red',label = '3. PC ('
               + str(0)[0:4] + '%)')
else:
    plt.quiver(*origin, pca['v_k'][0,2],
               pca['v_k'][1,2], pca['v_k'][2,2], color = 'red', label = '3. PC ('
               + str(100*rho[2])[0:4] + '%)')
p = ax.scatter3D(raw_data[0], raw_data[1], raw_data[2], color = 'black', label = 'Data points')

# Put a legend below current axis
ax.legend()

plt.show()

#%% Karcher mean for simulated data on geodesic curve

data_num = 10
p1 = np.array([0,0,1])
p2 = np.array([np.sqrt(2)/2, np.sqrt(2)/2,0])
t = np.linspace(-0.5,0.5,100)

curve = sphere_statistics(data_type = "Geodesic_curve", n = data_num, p = p1, q = p2)
data = curve.get_data()

mean_val = curve.intrinsic_mean_gradient_search(tau = 1)

pga = curve.pga(mean_val[:,-1])

geodesic_line1 = curve.geodesic_line(t, mean_val[:,-1],pga['v_k_exp'][:,0])
geodesic_line2 = curve.geodesic_line(t, mean_val[:,-1],pga['v_k_exp'][:,1])

# Compute variance explained by principal components
rho = (pga['lambda']*pga['lambda']) / (pga['lambda']*pga['lambda']).sum()
rho = rho[:-1] 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal geodesics');
plt.xlabel('Principal geodesics');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.ylim([0,1.05])
plt.show()

data_tangent, geodesic_line_ts1, geodesic_line_ts2, mean_val_ts = curve.tangent_space(np.array([1/2,1/2,np.sqrt(2)/2]), data, 
                                                                geodesic_line1, geodesic_line2,
                                                                mean_val)

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.scatter(data_tangent[0], data_tangent[1], color = 'black', 
            label = 'Data points')
plt.plot(geodesic_line_ts1[0], geodesic_line_ts1[1], 
         label = '1. PG ('+ str(100*rho[0])[0:4] + '%)', color = 'blue')
if rho[1]<0.001:
    plt.plot(geodesic_line_ts2[0], geodesic_line_ts2[1], 
         label = '2. PG ('+ str(0) + '%)', color = 'orange')
else:
    plt.plot(geodesic_line_ts2[0], geodesic_line_ts2[1], 
         label = '2. PG ('+ str(100*rho[1])[:4] + '%)', color = 'orange')
p = plt.scatter(mean_val_ts[0,-1], mean_val_ts[1,-1], color = 'blue',
                s = 100*np.ones(mean_val.shape[1]), label = 'Karcher mean',
            alpha = 0.4)

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend()

# Plotting the data 
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')

p = ax.scatter3D(data[0], data[1], data[2], color = 'black', label = 'Data points')

p = ax.scatter3D(mean_val[0,-1], mean_val[1,-1], mean_val[2,-1],s=100, color = 'blue',
                 label = 'Karcher mean')

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

ax.plot(geodesic_line1[0,:], geodesic_line1[1,:], geodesic_line1[2,:], 
        label = '1. PG (' + str(100*rho[0])[0:4] + '%)')
if rho[1]<0.001:
    ax.plot(geodesic_line2[0,:], geodesic_line2[1,:], geodesic_line2[2,:], 
            label = '2. PG (' + str(0) + '%)')
else:
    ax.plot(geodesic_line2[0,:], geodesic_line2[1,:], geodesic_line2[2,:], 
            label = '2. PG (' + str(100*rho[1])[0:4] + '%)')

# Put a legend below current axis
ax.legend()

plt.show()

pca = curve.pca(mean_val[:,-1])

# Compute variance explained by principal components
rho = (pca['sigma']*pca['sigma']) / (pca['sigma']*pca['sigma']).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# Plotting the data 
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

origin = pca['mean'][0], pca['mean'][1], pca['mean'][2]
p = ax.scatter3D(pca['mean'][0], pca['mean'][1], pca['mean'][2], color = 'blue',
                label = 'Arithmetic Mean', s = 100,
                alpha = 0.4)
plt.quiver(*origin, pca['v_k'][0,0],
           pca['v_k'][1,0], pca['v_k'][2,0], color = 'blue', label = '1. PC (' 
           + str(100*rho[0])[0:4] + '%)')
if rho[1]<0.001:
    plt.quiver(*origin, pca['v_k'][0,1],
               pca['v_k'][1,1], pca['v_k'][2,1], color = 'orange', label = '2. PC ('
               + str(0) + '%)')
else:
    plt.quiver(*origin, pca['v_k'][0,1],
               pca['v_k'][1,1], pca['v_k'][2,1], color = 'orange', label = '2. PC ('
               + str(100*rho[1])[0:4] + '%)')
if rho[2] < 0.001:    
    plt.quiver(*origin, pca['v_k'][0,2],
               pca['v_k'][1,2], pca['v_k'][2,2], color = 'red',label = '3. PC ('
               + str(0)[0:4] + '%)')
else:
    plt.quiver(*origin, pca['v_k'][0,2],
               pca['v_k'][1,2], pca['v_k'][2,2], color = 'red', label = '3. PC ('
               + str(100*rho[2])[0:4] + '%)')
p = ax.scatter3D(data[0], data[1], data[2], color = 'black', label = 'Data points')

# Put a legend below current axis
ax.legend()

plt.show()

#%% Karcher mean for simulated data on geodesic ball

data_num = 10
p1 = np.array([0,0,1])
radius = 1

ball = sphere_statistics(data_type = "Geodesic_ball", n = data_num, p = p1, r = radius)
data = ball.get_data()

mean_val = ball.intrinsic_mean_gradient_search(tau = 1)

pga = ball.pga(mean_val[:,-1])

geodesic_line1 = ball.geodesic_line(t, mean_val[:,-1],pga['v_k_exp'][:,0])
geodesic_line2 = ball.geodesic_line(t, mean_val[:,-1],pga['v_k_exp'][:,1])

# Compute variance explained by principal components
rho = (pga['lambda']*pga['lambda']) / (pga['lambda']*pga['lambda']).sum() 
rho = rho[:-1] 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal geodesics');
plt.xlabel('Principal geodesics');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.ylim([0,1.05])
plt.show()

data_tangent, geodesic_line_ts1, geodesic_line_ts2, mean_val_ts = ball.tangent_space(np.array([0,0,1]), data, 
                                                                geodesic_line1, geodesic_line2,
                                                                mean_val)

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.scatter(data_tangent[0], data_tangent[1], color = 'black', 
            label = 'Data points')
plt.plot(geodesic_line_ts1[0], geodesic_line_ts1[1], 
         label = '1. PG ('+ str(100*rho[0])[0:4] + '%)', color = 'blue')
if rho[1]<0.001:
    plt.plot(geodesic_line_ts2[0], geodesic_line_ts2[1], 
         label = '2. PG ('+ str(0) + '%)', color = 'orange')
else:
    plt.plot(geodesic_line_ts2[0], geodesic_line_ts2[1], 
         label = '2. PG ('+ str(100*rho[1])[:4] + '%)', color = 'orange')
p = plt.scatter(mean_val_ts[0,-1], mean_val_ts[1,-1], color = 'blue',
                s = 100*np.ones(mean_val.shape[1]), label = 'Karcher mean',
            alpha = 0.4)

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend()

# Plotting the data 
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')

p = ax.scatter3D(data[0], data[1], data[2], color = 'black', label = 'Data points')

p = ax.scatter3D(mean_val[0,-1], mean_val[1,-1], mean_val[2,-1],s=100, color = 'blue',
                 label = 'Karcher mean')

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

ax.plot(geodesic_line1[0,:], geodesic_line1[1,:], geodesic_line1[2,:], 
        label = '1. PG (' + str(100*rho[0])[0:4] + '%)')
if rho[1]<0.001:
    ax.plot(geodesic_line2[0,:], geodesic_line2[1,:], geodesic_line2[2,:], 
            label = '2. PG (' + str(0) + '%)')
else:
    ax.plot(geodesic_line2[0,:], geodesic_line2[1,:], geodesic_line2[2,:], 
            label = '2. PG (' + str(100*rho[1])[0:4] + '%)')

# Put a legend below current axis
ax.legend()

plt.show()

pca = ball.pca(mean_val[:,-1])

# Compute variance explained by principal components
rho = (pca['sigma']*pca['sigma']) / (pca['sigma']*pca['sigma']).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# Plotting the data 
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

origin = pca['mean'][0], pca['mean'][1], pca['mean'][2]
p = ax.scatter3D(pca['mean'][0], pca['mean'][1], pca['mean'][2], color = 'blue',
                label = 'Arithmetic Mean', s = 100,
                alpha = 0.4)
plt.quiver(*origin, pca['v_k'][0,0],
           pca['v_k'][1,0], pca['v_k'][2,0], color = 'blue', label = '1. PC (' 
           + str(100*rho[0])[0:4] + '%)')
if rho[1]<0.001:
    plt.quiver(*origin, pca['v_k'][0,1],
               pca['v_k'][1,1], pca['v_k'][2,1], color = 'orange', label = '2. PC ('
               + str(0) + '%)')
else:
    plt.quiver(*origin, pca['v_k'][0,1],
               pca['v_k'][1,1], pca['v_k'][2,1], color = 'orange', label = '2. PC ('
               + str(100*rho[1])[0:4] + '%)')
if rho[2] < 0.001:    
    plt.quiver(*origin, pca['v_k'][0,2],
               pca['v_k'][1,2], pca['v_k'][2,2], color = 'red',label = '3. PC ('
               + str(0)[0:4] + '%)')
else:
    plt.quiver(*origin, pca['v_k'][0,2],
               pca['v_k'][1,2], pca['v_k'][2,2], color = 'red', label = '3. PC ('
               + str(100*rho[2])[0:4] + '%)')
p = ax.scatter3D(data[0], data[1], data[2], color = 'black', label = 'Data points')

# Put a legend below current axis
ax.legend()

plt.show()





