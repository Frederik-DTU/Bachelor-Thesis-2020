# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 00:38:41 2020

@author: Frederik
"""

import numpy as np
from sphere_statistics import sphere_statistics
from sklearn.cluster import KMeans

from scipy.io import loadmat
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from circle_optimization import *

import os
import platform
import sys

#%% Plots and loading
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
data = knee_sphere['data']

n = data.shape[1] #Number of data points

# Checking NaN's or that the vectors aren't unit-length with a precision of eps
eps = 0.01
error_index = list()
for j in range(0, n):
    if abs(1-np.linalg.norm(data[:,j]))>eps:
        error_index.append(j)
        print("Data index, " + str(j) + ", is not unit length or contains NaN")

if not error_index:
    print("\nAll elements are unit length and don't contain any missing values!\n")

#Plot sphere
fig = plt.figure()
ax = plt.axes(projection='3d')

r = 1
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
x = r*sin(phi)*cos(theta)
y = r*sin(phi)*sin(theta)
z = r*cos(phi)

ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='c', alpha=1, linewidth=0)

plt.show()

# Plotting the data 
fig = plt.figure()
ax = plt.axes(projection='3d')
col_val = np.linspace(start = 0, stop = 0, num = n) #Set start=stop for same color

p = ax.scatter3D(data[0], data[1], data[2], c = col_val);

#fig.colorbar(p)

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

plt.show()

#%% Periodes plots

periods = [53, 105, 153, 217, 278]
periode_data = [data[:,0:periods[0]], data[:,periods[0]:periods[1]],
                data[:,periods[1]:periods[2]], data[:,periods[2]:periods[3]],
                data[:,periods[3]:periods[4]], data[:,periods[4]:]]
for i in range(0,6):
    fig = plt.figure(figsize = (5,5))
    ax = plt.axes(projection='3d')

    p = ax.scatter3D(periode_data[i][0], periode_data[i][1], periode_data[i][2], 
                     label = 'Periode ' + str(i+1));

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
    
#%% Kmeans clustering

K = 4
clusters = []
sphere_objects = [None]*6
for i in range(0,6):
    sphere_objects[i] = sphere_statistics(data_type = "Own", data = periode_data[i],
                                       n = periode_data[i].shape[1])

for i in range(0,6):
    
    kmeans_rm = sphere_objects[i].k_means_euclidean(K, seed = 1)

    data_clusters = [[] for i in range(0,K)]

    for k in range(0,K):
        data_clusters[k] =kmeans_rm['Data Clusters'][k]
        
    clusters.append(data_clusters)

    # Plotting the data 
    fig = plt.figure(figsize = (5,5))
    ax = plt.axes(projection='3d')

    for k in range(0,K):
        ax.scatter3D(data_clusters[k][0], data_clusters[k][1], data_clusters[k][2],
                     label = 'Cluster' + str(k+1));

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
    
#%% Kmeans PCA

data_objects = [[[] for k in range(0,K)] for i in range(0,6)]
karcher_means = [[[] for k in range(0,K)] for i in range(0,6)]
pca = [[[] for k in range(0,K)] for i in range(0,6)]
variance_pca1 = []
variance_pca2 = []

for i in range(0,6):
    for k in range(0,K):
        data_objects[i][k] = sphere_statistics(data_type = "Own", 
                                            data =  clusters[i][k], 
                                            n = clusters[i][k].shape[1])
        
        karcher_means[i][k] = data_objects[i][k].intrinsic_mean_gradient_search(tau=1)[:,-1]
        
        pca[i][k] = data_objects[i][k].pca(karcher_means[i][k])
        test = (pca[i][k]['sigma']*pca[i][k]['sigma']) / (pca[i][k]['sigma']*pca[i][k]['sigma']).sum() 
        variance_pca1.append(test[0])
        variance_pca2.append(test[1])
    
#%% Clustering in each periode

K = 4
clusters = []
sphere_objects = [None]*6
for i in range(0,6):
    sphere_objects[i] = sphere_statistics(data_type = "Own", data = periode_data[i],
                                       n = periode_data[i].shape[1])

for i in range(0,6):
    
    kmeans_rm = sphere_objects[i].k_means_riemannian_manifold(K, seed = 1)

    data_clusters = [[] for i in range(0,K)]

    for k in range(0,K):
        data_clusters[k] =kmeans_rm['Data Clusters'][k]
        
    clusters.append(data_clusters)

    # Plotting the data 
    fig = plt.figure(figsize = (5,5))
    ax = plt.axes(projection='3d')

    for k in range(0,K):
        ax.scatter3D(data_clusters[k][0], data_clusters[k][1], data_clusters[k][2],
                     label = 'Cluster' + str(k+1));

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

#%% PGA

karcher_means = [[] for i in range(0,K)]
pga = [[] for i in range(0,K)]
geodesic_lines = [[] for i in range(0,K)]
variance_pga1 = []
variance_pga2 = []
t = np.linspace(-0.25,0.25, 100)


data_objects = [[[] for k in range(0,K)] for i in range(0,6)]
karcher_means = [[[] for k in range(0,K)] for i in range(0,6)]
pga = [[[] for k in range(0,K)] for i in range(0,6)]
geodesic_lines = [[[] for k in range(0,K)] for i in range(0,6)]

for i in range(0,6):
    for k in range(0,K):
        data_objects[i][k] = sphere_statistics(data_type = "Own", 
                                            data =  clusters[i][k], 
                                            n = clusters[i][k].shape[1])
        
        karcher_means[i][k] = data_objects[i][k].intrinsic_mean_gradient_search(tau=1)[:,-1]
        
        pga[i][k] = data_objects[i][k].pga(karcher_means[i][k])
        geodesic_lines[i][k] = [data_objects[i][k].geodesic_line(t, karcher_means[i][k],
                                                                 pga[i][k]['v_k_exp'][:,0]),
                          data_objects[i][k].geodesic_line(t, karcher_means[i][k],
                                                           pga[i][k]['v_k_exp'][:,1])]
        test = (pga[i][k]['lambda']*pga[i][k]['lambda']) / (pga[i][k]['lambda']*pga[i][k]['lambda']).sum()
        variance_pga1.append(test[0])
        variance_pga2.append(test[1])

for i in range(0,6):
    # Plotting the data 
    fig = plt.figure(figsize = (5,5))
    ax = plt.axes(projection='3d')
    
    for k in range(0,K):
        ax.scatter3D(clusters[i][k][0], clusters[i][k][1], clusters[i][k][2],
                     label = 'Cluster' + str(k+1));
        ax.scatter3D(karcher_means[i][k][0], karcher_means[i][k][1],
                 karcher_means[i][k][2], label = 'Karcher mean for cluster ' + str(k+1),
                 s = 300);
        ax.plot(geodesic_lines[i][k][0][0,:], geodesic_lines[i][k][0][1,:],
                geodesic_lines[i][k][0][2,:], 
                label = 'First Principal Component for cluster ' + str(k+1))
    
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

clusters_ts = [[[] for i in range(0,K)] for j in range(0,6)]
geodesic_lines1_ts = [[[] for i in range(0,K)] for j in range(0,6)]
geodesic_lines2_ts = [[[] for i in range(0,K)] for j in range(0,6)]
optimal_circle_ts = [[[] for i in range(0,K)] for j in range(0,6)]

for i in range(0,6):
    for k in range(0,K):
        clusters_ts[i][k], geodesic_lines1_ts[i][k], geodesic_lines2_ts[i][k], optimal_circle_ts[i][k] = knee_data.tangent_space(karcher_means[i][k], 
                                               clusters[i][k],
                                               geodesic_lines[i][k][0],
                                               geodesic_lines[i][k][1],
                                               param_rot)
    

for i in range(0,6):
    for k in range(0,K):    
        fig = plt.figure(figsize = (6,5))
        ax = plt.subplot(111)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        
        plt.scatter(clusters_ts[i][k][0], clusters_ts[i][k][1] ,
                    label = 'Cluster' + str(k+1), color = 'Black', s = 3)
        plt.plot(geodesic_lines1_ts[i][k][0], geodesic_lines1_ts[i][k][1] ,
                    label = '1. PG (' + str(100*variance_pga1[i*K+k])[0:4] + '%)')
        if variance_pga2[i*K+k]<0.001:
            plt.plot(geodesic_lines2_ts[i][k][0], geodesic_lines2_ts[i][k][1] ,
                     label = '2. PG (' + str(0) + '%)')
        else:
            plt.plot(geodesic_lines2_ts[i][k][0], geodesic_lines2_ts[i][k][1] ,
                     label = '2. PG (' + str(100*variance_pga2[i*K+k])[0:4] + '%)')
        plt.plot(optimal_circle_ts[i][k][0], optimal_circle_ts[i][k][1] ,
                    label = 'Optimal Circle' + str(k+1))
            
        p = plt.scatter(0,0, color = 'Blue',
                        s = 100*np.ones(mean_val.shape[1]),
                    alpha = 0.4, label = 'Karcher mean')
        
        # Shrink current axis by 20%
        box = ax.get_position()
        
        # Put a legend to the right of the current axis
        ax.legend(loc='upper right')
    
#%% Rotation of data to north pole

rot_clusters = [[[] for k in range(0,K)] for i in range(0,6)]
rot_geodesic_lines1 = [[[] for k in range(0,K)] for i in range(0,6)]
rot_geodesic_lines2 = [[[] for k in range(0,K)] for i in range(0,6)]
rot_optimal_circle = [[[] for k in range(0,K)] for i in range(0,6)]

for i in range(0,6):
    for k in range(0,K):
        rot_optimal_circle[i][k] = np.array([param_rot[0], param_rot[1], param_rot[2]])
        rot_clusters[i][k] = data_objects[i][k].rotate_data_to_north_pole(karcher_means[i][k], 
                                                                    clusters[i][k],
                                                                    clusters[i][k].shape[1])
        rot_geodesic_lines1[i][k] = data_objects[i][k].rotate_data_to_north_pole(karcher_means[i][k], 
                                                                    geodesic_lines[i][k][0],
                                                                    geodesic_lines[i][k][0].shape[1])
        rot_geodesic_lines2[i][k] = data_objects[i][k].rotate_data_to_north_pole(karcher_means[i][k], 
                                                                    geodesic_lines[i][k][1],
                                                                    geodesic_lines[i][k][1].shape[1])
        rot_optimal_circle[i][k] = data_objects[i][k].rotate_data_to_north_pole(karcher_means[i][k], 
                                                                    rot_optimal_circle[i][k],
                                                                    rot_optimal_circle[i][k].shape[1])
        
#%%Plotting rotations in 3D    

# Plotting the data 

for i in range(0,6):
    for k in range(0,K):
        fig = plt.figure(figsize = (5,5))
        ax = plt.axes(projection='3d')
        ax.scatter3D(rot_clusters[i][k][0], rot_clusters[i][k][1],
                     rot_clusters[i][k][2], label = 'Cluster ' + str(k+1));
        ax.scatter3D(0, 0,
                     1, label = 'Karcher mean for cluster ' + str(k+1),
                     s = 300);
        ax.plot(rot_geodesic_lines1[i][k][0,:], rot_geodesic_lines1[i][k][1,:], rot_geodesic_lines1[i][k][2,:], 
            label = 'First Principal Component for cluster ' + str(k+1))
        ax.plot(rot_geodesic_lines2[i][k][0,:], rot_geodesic_lines2[i][k][1,:], rot_geodesic_lines2[i][k][2,:], 
            label = 'Second Principal Component for cluster ' + str(k+1))
        ax.plot(rot_optimal_circle[i][k][0,:], rot_optimal_circle[i][k][1,:], 
                rot_optimal_circle[i][k][2,:], 
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

#%%Plotting rotations in 2D    

# Plotting the data 
for i in range(0,6):
    for k in range(0,K):
        fig = plt.figure(figsize = (5,5))
        ax = plt.axes()
    
        ax.scatter(rot_clusters[i][k][0], rot_clusters[i][k][1], label = 'Cluster ' + str(k+1));
        ax.scatter(0, 0, label = 'Karcher mean for cluster ' + str(k+1),
                     s = 300);
        ax.plot(rot_geodesic_lines1[i][k][0,:], rot_geodesic_lines1[i][k][1,:], 
            label = 'First Principal Component for cluster ' + str(k+1))
        ax.plot(rot_geodesic_lines2[i][k][0,:], rot_geodesic_lines2[i][k][1,:], 
            label = 'Second Principal Component for cluster ' + str(k+1))
        ax.plot(rot_optimal_circle[i][k][0,:], rot_optimal_circle[i][k][1,:], 
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
    
#%% Variance

fig = plt.figure(figsize = (5,5))
ax = plt.subplot(111)
plt.plot(variance_pca1, '-*', color = 'blue', label = 'Variance described by 1. PC') 
plt.plot(variance_pga1, '-*', color = 'orange', label = 'Variance described by 1. PG')
plt.xlabel('Cluster')
plt.ylabel('Fraction of variance')
ax.legend(loc = 'lower left')

fig = plt.figure(figsize = (5,5))
ax = plt.subplot(111)
plt.plot(variance_pca2, '-*', color = 'blue', label = 'Variance described by 2. PC') 
plt.plot(variance_pga2, '-*', color = 'orange', label = 'Variance described by 2. PG')
plt.xlabel('Cluster')
plt.ylabel('Fraction of variance')
ax.legend(loc = 'upper left')


    
    
    
    
    



