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

from sklearn.cluster import KMeans

from circle_optimization import *

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

#%% Kmeans

K = 4
kmeans = knee_data.k_means_euclidean(K, seed = 50)

# Plotting the data 
fig = plt.figure(figsize = (5,5))
ax = plt.axes(projection='3d')

for k in range(0,K):
    ax.scatter3D(kmeans['Data Clusters'][k][0], kmeans['Data Clusters'][k][1],
                 kmeans['Data Clusters'][k][2], label = 'Cluster ' + str(k+1));

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

kmeans_ts = [[] for k in range(0,K)]
for k in range(0,K):
    kmeans_ts[k] = knee_data.tangent_space(mean_val[:,-1], kmeans['Data Clusters'][k])[0]

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val.shape[1]) #Set start=stop for same color

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-3, 3)
plt.ylim(-3, 3)

for k in range(0,K):
    plt.scatter(kmeans_ts[k][0], kmeans_ts[k][1] ,
                label = 'Cluster' + str(k+1))
p = plt.scatter(0,0, color = 'Blue',
                s = 100*np.ones(mean_val.shape[1]),
            alpha = 0.4, label = 'Karcher mean')

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend(loc='upper right')

#%% Mean and PCA locally

data_objects = [[] for i in range(0,K)]
karcher_means = [[] for i in range(0,K)]
pca = [[] for i in range(0,K)]
variance_pca = [[] for i in range(0,K)]

for k in range(0,K):
    data_objects[k] = sphere_statistics(data_type = "Own", 
                                        data = kmeans['Data Clusters'][k], 
                                        n = kmeans['Data Clusters'][k].shape[1])
    karcher_means[k] = data_objects[k].intrinsic_mean_gradient_search(tau=1)[:,-1]
    
    pca[k] = data_objects[k].pca(karcher_means[k])
    variance_pca[k] = (pca[k]['sigma']*pca[k]['sigma']) / (pca[k]['sigma']*pca[k]['sigma']).sum() 
    
#%% K-means RM
K = 4
kmeans_rm = knee_data.k_means_riemannian_manifold(K, seed = 50)

# Plotting the data 
fig = plt.figure(figsize = (5,5))
ax = plt.axes(projection='3d')

for k in range(0,K):
    ax.scatter3D(kmeans_rm['Data Clusters'][k][0], kmeans_rm['Data Clusters'][k][1],
                 kmeans_rm['Data Clusters'][k][2], label = 'Cluster ' + str(k+1));

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

kmeans_ts = [[] for k in range(0,K)]
for k in range(0,K):
    kmeans_ts[k] = knee_data.tangent_space(mean_val[:,-1], kmeans_rm['Data Clusters'][k])[0]

col_val = np.linspace(start = 0.5, stop = 1, num = mean_val.shape[1]) #Set start=stop for same color

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-3, 3)
plt.ylim(-3, 3)

for k in range(0,K):
    plt.scatter(kmeans_ts[k][0], kmeans_ts[k][1] ,
                label = 'Cluster' + str(k+1))
p = plt.scatter(0,0, color = 'Blue',
                s = 100*np.ones(mean_val.shape[1]),
            alpha = 0.4, label = 'Karcher mean')

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend(loc='upper right')


#%% Karcher mean and pga locally

data_objects = [[] for i in range(0,K)]
karcher_means = [[] for i in range(0,K)]
pga = [[] for i in range(0,K)]
geodesic_lines = [[] for i in range(0,K)]
variance_pga = [[] for i in range(0,K)]
t = np.linspace(-0.25,0.25, 100)

for k in range(0,K):
    data_objects[k] = sphere_statistics(data_type = "Own", 
                                        data = kmeans_rm['Data Clusters'][k], 
                                        n = kmeans_rm['Data Clusters'][k].shape[1])
    karcher_means[k] = data_objects[k].intrinsic_mean_gradient_search(tau=1)[:,-1]
    
    pga[k] = data_objects[k].pga(karcher_means[k])
    variance_pga[k] = (pga[k]['lambda']*pga[k]['lambda']) / (pga[k]['lambda']*pga[k]['lambda']).sum() 
    geodesic_lines[k] = [data_objects[k].geodesic_line(t, karcher_means[k],pga[k]['v_k_exp'][:,0]),
                      data_objects[k].geodesic_line(t, karcher_means[k],pga[k]['v_k_exp'][:,1])]

#%% Karcher mean plot

# Plotting the data 
fig = plt.figure(figsize = (5,5))
ax = plt.axes(projection='3d')

for k in range(0,K):
    ax.scatter3D(kmeans_rm['Data Clusters'][k][0], kmeans_rm['Data Clusters'][k][1],
                 kmeans_rm['Data Clusters'][k][2], label = 'Cluster ' + str(k+1));
    ax.scatter3D(karcher_means[k][0], karcher_means[k][1],
                 karcher_means[k][2], label = 'Karcher mean for cluster ' + str(k+1),
                 s = 300);

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

#%% PGA plot

# Plotting the data 
fig = plt.figure(figsize = (5,5))
ax = plt.axes(projection='3d')

for k in range(0,K):
    ax.scatter3D(kmeans_rm['Data Clusters'][k][0], kmeans_rm['Data Clusters'][k][1],
                 kmeans_rm['Data Clusters'][k][2], label = 'Cluster ' + str(k+1));
    ax.scatter3D(karcher_means[k][0], karcher_means[k][1],
                 karcher_means[k][2], label = 'Karcher mean for cluster ' + str(k+1),
                 s = 300);
    ax.plot(geodesic_lines[k][0][0,:], geodesic_lines[k][0][1,:], geodesic_lines[k][0][2,:], 
        label = 'First Principal Component for cluster ' + str(k+1))
    ax.plot(geodesic_lines[k][1][0,:], geodesic_lines[k][1][1,:], geodesic_lines[k][1][2,:],
        label = 'Second Principal Component for cluster ' + str(k+1))

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

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()

plt.show()

#%%Plot in tangent space

clusters_ts = [[] for i in range(0,K)]
geodesic_lines1_ts = [[] for i in range(0,K)]
geodesic_lines2_ts = [[] for i in range(0,K)]
optimal_circle_ts = [[] for i in range(0,K)]

for k in range(0,K):
    clusters_ts[k], geodesic_lines1_ts[k], geodesic_lines2_ts[k], optimal_circle_ts[k] = knee_data.tangent_space(karcher_means[k], 
                                           kmeans_rm['Data Clusters'][k],
                                           geodesic_lines[k][0],
                                           geodesic_lines[k][1],
                                           param_rot)
    

for k in range(0,K):    
    fig = plt.figure(figsize = (6,5))
    ax = plt.subplot(111)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    plt.scatter(clusters_ts[k][0], clusters_ts[k][1] ,
                label = 'Cluster' + str(k+1), color = 'Black', s = 1)
    plt.plot(geodesic_lines1_ts[k][0], geodesic_lines1_ts[k][1] ,
                label = '1. PG (' + str(100*variance_pga[k][0])[0:4] + '%)')
    if variance_pga[k][1]<0.001:
        plt.plot(geodesic_lines2_ts[k][0], geodesic_lines2_ts[k][1] ,
                 label = '2. PG (' + str(0) + '%)')
    else:
        plt.plot(geodesic_lines2_ts[k][0], geodesic_lines2_ts[k][1] ,
                 label = '2. PG (' + str(100*variance_pga[k][1])[0:4] + '%)')
    plt.plot(optimal_circle_ts[k][0], optimal_circle_ts[k][1] ,
                label = 'Optimal Circle' + str(k+1))
        
    p = plt.scatter(0,0, color = 'Blue',
                    s = 100*np.ones(mean_val.shape[1]),
                alpha = 0.4, label = 'Karcher mean')
    
    # Shrink current axis by 20%
    box = ax.get_position()
    
    # Put a legend to the right of the current axis
    ax.legend(loc='upper right')


#%% Rotation of data to north pole

rot_clusters = [[] for i in range(0,K)]
rot_geodesic_lines1 = [[] for i in range(0,K)]
rot_geodesic_lines2 = [[] for i in range(0,K)]
rot_optimal_circle = [[] for i in range(0,K)]

for k in range(0,K):
    rot_optimal_circle[k] = np.array([param_rot[0], param_rot[1], param_rot[2]])
    rot_clusters[k] = data_objects[k].rotate_data_to_north_pole(karcher_means[k], 
                                                                kmeans_rm['Data Clusters'][k],
                                                                kmeans_rm['Data Clusters'][k].shape[1])
    rot_geodesic_lines1[k] = data_objects[k].rotate_data_to_north_pole(karcher_means[k], 
                                                                geodesic_lines[k][0],
                                                                geodesic_lines[k][0].shape[1])
    rot_geodesic_lines2[k] = data_objects[k].rotate_data_to_north_pole(karcher_means[k], 
                                                                geodesic_lines[k][1],
                                                                geodesic_lines[k][1].shape[1])
    rot_optimal_circle[k] = data_objects[k].rotate_data_to_north_pole(karcher_means[k], 
                                                                rot_optimal_circle[k],
                                                                rot_optimal_circle[k].shape[1])
    
#%%Plotting rotations in 3D    

# Plotting the data 

for k in range(0,K):
    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    ax.scatter3D(rot_clusters[k][0], rot_clusters[k][1],
                 rot_clusters[k][2], label = 'Cluster ' + str(k+1));
    ax.scatter3D(0, 0,
                 1, label = 'Karcher mean for cluster ' + str(k+1),
                 s = 300);
    ax.plot(rot_geodesic_lines1[k][0,:], rot_geodesic_lines1[k][1,:], rot_geodesic_lines1[k][2,:], 
        label = 'First Principal Component for cluster ' + str(k+1))
    ax.plot(rot_geodesic_lines2[k][0,:], rot_geodesic_lines2[k][1,:], rot_geodesic_lines2[k][2,:], 
        label = 'Second Principal Component for cluster ' + str(k+1))
    ax.plot(rot_optimal_circle[k][0,:], rot_optimal_circle[k][1,:], rot_optimal_circle[k][2,:], 
        label = 'Optimal Circle', c = 'blue')
    #ax.plot(geodesic_lines[k][1][0,:], geodesic_lines[k][1][1,:], geodesic_lines[k][1][2,:],
    #    label = 'Second Principal Component for cluster ' + str(k+1))


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

#%%Plotting rotations in 2D    

# Plotting the data 
for k in range(0,K):
    fig = plt.figure(figsize = (8,8))
    ax = plt.axes()

    ax.scatter(rot_clusters[k][0], rot_clusters[k][1], label = 'Cluster ' + str(k+1));
    ax.scatter(0, 0, label = 'Karcher mean for cluster ' + str(k+1),
                 s = 300);
    ax.plot(rot_geodesic_lines1[k][0,:], rot_geodesic_lines1[k][1,:], 
        label = 'First Principal Component for cluster ' + str(k+1))
    ax.plot(rot_geodesic_lines2[k][0,:], rot_geodesic_lines2[k][1,:], 
        label = 'Second Principal Component for cluster ' + str(k+1))
    ax.plot(rot_optimal_circle[k][0,:], rot_optimal_circle[k][1,:], 
        label = 'Optimal Circle', c = 'blue')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    plt.tight_layout()
    
    # Adding a sphere (S2) to the data
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    theta = np.mgrid[0.0:2.0*pi:100j]
    x = r*cos(theta)
    y = r*sin(theta)
    z = r*cos(phi)
    
    ax.plot(
        x, y, z,  color='c')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    ax.legend()
    
    plt.show()

#%% Haussdorf

haussdorf_dist = knee_data.haussdorf_distance(kmeans['Data Clusters'],
                                              kmeans_rm['Data Clusters'])

#%% Variance

plot_pca = [[] for j in range(0,3)]
plot_pga = [[] for j in range(0,3)]
for j in range(0,3):
    for k in range(0,K):
        plot_pca[j].append(variance_pca[k][j])
        plot_pga[j].append(variance_pga[k][j])

fig = plt.figure(figsize = (5,5))
ax = plt.subplot(111)
plt.plot(plot_pca[0], '-*', color = 'blue', label = 'Variance described by 1. PC') 
plt.plot(plot_pga[0], '-*', color = 'orange', label = 'Variance described by 1. PG')
plt.xlabel('Cluster')
plt.ylabel('Fraction of variance')
ax.legend(loc = 'upper left')

fig = plt.figure(figsize = (5,5))
ax = plt.subplot(111)
plt.plot(plot_pca[1], '-*', color = 'blue', label = 'Variance described by 2. PC') 
plt.plot(plot_pga[1], '-*', color = 'orange', label = 'Variance described by 2. PG')
plt.xlabel('Cluster')
plt.ylabel('Fraction of variance')
ax.legend(loc = 'upper left')


