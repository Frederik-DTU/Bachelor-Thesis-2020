import numpy as np

from scipy.io import loadmat
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sphere_statistics import sphere_statistics

import os
import platform
import sys

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
fig = plt.figure(figsize=(6,5))
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

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()

plt.show()

# Plotting the data 
fig = plt.figure(figsize=(5,5))
ax = plt.axes(projection='3d')
col_val = np.linspace(start = 0, stop = 0, num = n) #Set start=stop for same color

p = ax.scatter3D(data[0], data[1], data[2], color = 'black');

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

ax.legend()

plt.show()

knee_data = sphere_statistics(data_type = "Own", data = data, n = n)
data_tangent = knee_data.tangent_space(np.array([0,0,1]), data)

fig = plt.figure(figsize = (6,5))
ax = plt.subplot(111)
plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.scatter(data_tangent[0][0], data_tangent[0][1], color = 'black', 
            label = 'Data points')
p = plt.scatter(0, 0, color = 'blue', s = 100, alpha = 0.4,
                label = 'North pole')

# Shrink current axis by 20%
box = ax.get_position()

# Put a legend to the right of the current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)

npole = np.array([0,0,1])

length_to_npole = np.zeros(n)
length_to_center = np.zeros(n-1)
length_between = np.zeros(n-1)

for i in range(0,n):
    length_to_npole[i] = np.arccos(np.dot(np.transpose(npole),data[:,i]))

fig = plt.figure()
plt.plot(np.linspace(start = 1, stop = n, num = n), length_to_npole)
plt.xlabel("Data number")
plt.ylabel("dist(p,x_i)")
plt.show()

for i in range(0,n-1):
    length_to_center[i] = np.linalg.norm(data[:,i+1]-data[:,1])
    length_between[i] = np.linalg.norm(data[:,i+1]-data[:,i])

fig = plt.figure()
plt.plot(np.linspace(start = 2, stop = n, num = n-1), length_to_center)
plt.show()

fig = plt.figure()
plt.plot(np.linspace(start = 2, stop = n, num = n-1), length_between)
plt.show()

for i in range(0,n-1):
    length_to_center[i] = np.arccos(np.dot(np.transpose(data[:,i+1]),data[:,1]))
    length_between[i] = np.arccos(np.dot(np.transpose(data[:,i+1]),data[:,i]))
    
np.argmin(length_to_center[40:60]) #53
np.argmin(length_to_center[80:120]) #105
np.argmin(length_to_center[140:170]) #153
np.argmin(length_to_center[180:250]) #217
np.argmin(length_to_center[250:300]) #278

periods = [53, 105, 105, 153, 217, 278]

fig = plt.figure()
plt.plot(np.linspace(start = 2, stop = n, num = n-1), length_to_center)
plt.xlabel("Data number")
plt.ylabel("dist(x_1,x_i)")

for i in periods: 
    plt.plot(i+2,length_to_center[i], marker = ".", color = "red", markersize=5)
plt.show()

#%% Tangent plot

p1 = np.array([0,0,1])
v2 = np.array([1,0,0])
v3 = np.array([1,-1,0])
p2 = p1*np.cos(np.linalg.norm(v2))+v2*np.sin(np.linalg.norm(v2))/np.linalg.norm(v2)
p3 = p1*np.cos(np.linalg.norm(v3))+v3*np.sin(np.linalg.norm(v3))/np.linalg.norm(v3)
#p2 = np.array([np.sqrt(2)/2,0,np.sqrt(2)/2])
#p3 = np.array([0,np.sqrt(2)/2,np.sqrt(2)/2])
t = np.linspace(0,1,100)

data = np.stack([p1,p2,p3], axis = 1)
tangent = sphere_statistics(data_type = "Own", data = data, n = 3)

geodesic1 = tangent.geodesic_line2(p1,p2)
geodesic2 = tangent.geodesic_line2(p1,p3)
geodesic3 = tangent.geodesic_line2(p2,p3)

data_tangent, geodesic1_ts, geodesic2_ts, geodesic3_ts = tangent.tangent_space(p1, data, 
                                                                geodesic1,
                                                                geodesic2,
                                                                geodesic3)
geodesic4_ts = np.zeros([3,100])
geodesic4_ts[0] = data_tangent[0,1]+(data_tangent[0,2]-data_tangent[0,1])*t
geodesic4_ts[1] = data_tangent[1,1]+(data_tangent[1,2]-data_tangent[1,1])*t
geodesic4_ts[2] = 0

#geodesic4 = np.zeros([3,100])
#for j in range(0,100):
#    geodesic3[:,j] = tangent.Exp_map(p1,geodesic3_ts[:,j], 1)

# Plotting the data 
fig = plt.figure(figsize=(5,6))
ax = plt.axes(projection='3d')

p = ax.scatter3D(data[0], data[1], data[2], color = 'black');
ax.plot(geodesic1[0], geodesic1[1], geodesic1[2], color = 'black');
ax.plot(geodesic2[0], geodesic2[1], geodesic2[2], color = 'black');
ax.plot(geodesic3[0], geodesic3[1], geodesic3[2], color = 'black');
ax.text(data[0][0], data[1][0]+0.1, data[2][0], '\u03bc')
ax.text(data[0][1]+0.1, data[1][1]+0.1, data[2][1]+0.1, 'y')
ax.text(data[0][2], data[1][2]-0.1, data[2][2]-0.1, 'x')
ax.text(geodesic1[0][50], geodesic1[1][50], geodesic1[2][50], '$||Log_\u03bc(y)||$')
ax.text(geodesic2[0][50]-0.5, geodesic2[1][50]-0.5, geodesic2[2][50]-0.1, '$||Log_\u03bc(x)||$')
ax.text(geodesic3[0][50]+0.1, geodesic3[1][50]-0.1, geodesic3[2][50]-0.1, '$||Log_x(y)||$')

#fig.colorbar(p)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,2])
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

fig = plt.figure(figsize = (5,5))
ax = plt.subplot(111)
plt.xlim(-2, 3)
plt.ylim(-2, 2)

plt.scatter(data_tangent[0], data_tangent[1], color = 'black', 
            label = 'Data points')
plt.annotate('\u03bc', (-0.2, 0))
plt.annotate('y', (data_tangent[0][1]+0.1, data_tangent[1][1]))
plt.annotate('x', (data_tangent[0][2]+0.1, data_tangent[1][2]))
plt.annotate('$||Log_\u03bc(y)||$', (data_tangent[0][1]/2-0.5, 0.2))
plt.annotate('$||Log_\u03bc(x)||$', (-0.6, data_tangent[1][2]/2))
plt.annotate('$||Log_\u03bc(x)-Log_\u03bc(y)||$', (data_tangent[0][1], data_tangent[1][2]+0.5))
plt.plot(geodesic1_ts[0], geodesic1_ts[1], 
         label = 'Geodesic line', color = 'blue')
plt.plot(geodesic2_ts[0], geodesic2_ts[1], 
         label = 'Geodesic line', color = 'blue')
plt.plot(geodesic4_ts[0], geodesic4_ts[1], 
         label = 'Geodesic line', color = 'blue')



